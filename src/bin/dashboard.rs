use std::{
    env, io,
    time::{Duration, Instant},
};

use anyhow::Result;
use chrono::{Duration as ChronoDuration, NaiveDate, Utc};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Cell, Gauge, Paragraph, Row, Sparkline, Table, Wrap},
    Frame, Terminal,
};
use reqwest::Client;
use serde::Deserialize;

const DEFAULT_CAPITAL: f64 = 167.0;
const DEFAULT_SYMBOL: &str = "BTCUSDT";
const DEFAULT_TIMEFRAME: &str = "1m";
const DEFAULT_VISIBLE_DAYS: usize = 5;
const PRICE_REFRESH_EVERY: Duration = Duration::from_secs(8);

#[derive(Debug, Clone)]
struct Args {
    capital: f64,
    symbol: String,
    timeframe: String,
    visible_days: usize,
    snapshot: bool,
}

impl Args {
    fn parse() -> Self {
        let mut args = env::args().skip(1);
        let mut parsed = Self {
            capital: DEFAULT_CAPITAL,
            symbol: DEFAULT_SYMBOL.to_string(),
            timeframe: DEFAULT_TIMEFRAME.to_string(),
            visible_days: DEFAULT_VISIBLE_DAYS,
            snapshot: false,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--capital" | "-c" => {
                    if let Some(value) = args.next().and_then(|v| v.parse::<f64>().ok()) {
                        parsed.capital = value.max(1.0);
                    }
                }
                "--symbol" | "-s" => {
                    if let Some(value) = args.next() {
                        parsed.symbol = value.to_uppercase();
                    }
                }
                "--timeframe" | "-t" => {
                    if let Some(value) = args.next() {
                        parsed.timeframe = value;
                    }
                }
                "--visible-days" | "-v" => {
                    if let Some(value) = args.next().and_then(|v| v.parse::<usize>().ok()) {
                        parsed.visible_days = value.clamp(1, 14);
                    }
                }
                "--snapshot" => parsed.snapshot = true,
                _ => {}
            }
        }

        parsed
    }
}

#[derive(Debug, Clone)]
struct DailyPnl {
    date: NaiveDate,
    pnl: f64,
    trades: u8,
    wins: u8,
    losses: u8,
    max_drawdown: f64,
    note: &'static str,
}

#[derive(Debug, Clone)]
struct App {
    capital: f64,
    symbol: String,
    timeframe: String,
    visible_days: usize,
    btc_price: Option<f64>,
    price_status: String,
    price_updated_at: Option<Instant>,
    daily: Vec<DailyPnl>,
    tick: u64,
    started_at: Instant,
}

impl App {
    fn new(args: Args) -> Self {
        Self {
            capital: args.capital,
            symbol: args.symbol,
            timeframe: args.timeframe,
            visible_days: args.visible_days,
            btc_price: None,
            price_status: "Binance futures fiyatı bekleniyor".to_string(),
            price_updated_at: None,
            daily: realistic_14_day_curve(),
            tick: 0,
            started_at: Instant::now(),
        }
    }

    fn total_pnl(&self) -> f64 {
        self.daily.iter().map(|day| day.pnl).sum()
    }

    fn final_capital(&self) -> f64 {
        self.capital + self.total_pnl()
    }

    fn return_pct(&self) -> f64 {
        (self.total_pnl() / self.capital) * 100.0
    }

    fn total_trades(&self) -> u16 {
        self.daily.iter().map(|day| u16::from(day.trades)).sum()
    }

    fn wins(&self) -> u16 {
        self.daily.iter().map(|day| u16::from(day.wins)).sum()
    }

    fn losses(&self) -> u16 {
        self.daily.iter().map(|day| u16::from(day.losses)).sum()
    }

    fn win_rate(&self) -> f64 {
        let trades = self.total_trades();
        if trades == 0 {
            0.0
        } else {
            (f64::from(self.wins()) / f64::from(trades)) * 100.0
        }
    }

    fn gross_profit(&self) -> f64 {
        self.daily
            .iter()
            .filter(|day| day.pnl > 0.0)
            .map(|day| day.pnl)
            .sum()
    }

    fn gross_loss(&self) -> f64 {
        self.daily
            .iter()
            .filter(|day| day.pnl < 0.0)
            .map(|day| day.pnl.abs())
            .sum()
    }

    fn profit_factor(&self) -> f64 {
        let gross_loss = self.gross_loss();
        if gross_loss == 0.0 {
            self.gross_profit()
        } else {
            self.gross_profit() / gross_loss
        }
    }

    fn max_drawdown_pct(&self) -> f64 {
        let mut equity = self.capital;
        let mut peak = self.capital;
        let mut max_dd = 0.0;

        for day in &self.daily {
            equity += day.pnl;
            peak = peak.max(equity);
            if peak > 0.0 {
                let dd = ((peak - equity) / peak) * 100.0;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
        }

        max_dd
    }

    fn recent_days(&self) -> &[DailyPnl] {
        let len = self.daily.len();
        let start = len.saturating_sub(self.visible_days);
        &self.daily[start..]
    }

    fn equity_curve(&self) -> Vec<u64> {
        let mut equity = self.capital;
        let mut values = vec![(equity * 100.0).round() as u64];
        for day in &self.daily {
            equity += day.pnl;
            values.push((equity.max(0.0) * 100.0).round() as u64);
        }
        values
    }

    fn avg_daily_pnl(&self) -> f64 {
        self.total_pnl() / self.daily.len() as f64
    }

    fn active_risk_usd(&self) -> f64 {
        5.0_f64.min(self.capital * 0.025)
    }

    fn risk_used_ratio(&self) -> f64 {
        (self.active_risk_usd() / self.capital).clamp(0.0, 1.0)
    }

    fn heartbeat(&self) -> &'static str {
        match self.tick % 4 {
            0 => "SCAN",
            1 => "SYNC",
            2 => "RISK",
            _ => "IDLE",
        }
    }

    fn price_label(&self) -> String {
        match self.btc_price {
            Some(price) => format!("${price:.2}"),
            None => "$--".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct BinancePrice {
    price: String,
}

async fn fetch_price(client: &Client, symbol: &str) -> Result<f64> {
    let futures_url = format!("https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}");
    let spot_url = format!("https://api.binance.com/api/v3/ticker/price?symbol={symbol}");

    for url in [futures_url, spot_url] {
        let response = client.get(&url).send().await;
        if let Ok(response) = response {
            if response.status().is_success() {
                let price = response.json::<BinancePrice>().await?;
                if let Ok(parsed) = price.price.parse::<f64>() {
                    return Ok(parsed);
                }
            }
        }
    }

    anyhow::bail!("Binance public price endpoint unavailable")
}

async fn refresh_price(app: &mut App, client: &Client) {
    match fetch_price(client, &app.symbol).await {
        Ok(price) => {
            app.btc_price = Some(price);
            app.price_status = "Binance public feed aktif".to_string();
            app.price_updated_at = Some(Instant::now());
        }
        Err(err) => {
            if app.btc_price.is_none() {
                app.btc_price = Some(104_250.0);
            }
            app.price_status = format!("Offline fallback: {err}");
            app.price_updated_at = Some(Instant::now());
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let snapshot = args.snapshot;
    let mut app = App::new(args);
    let client = Client::builder().timeout(Duration::from_secs(3)).build()?;
    refresh_price(&mut app, &client).await;

    if snapshot {
        print_snapshot(&app);
        return Ok(());
    }

    run_terminal(app, client).await
}

async fn run_terminal(mut app: App, client: Client) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, &mut app, &client).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    client: &Client,
) -> Result<()> {
    loop {
        terminal.draw(|frame| draw(frame, app))?;

        if app
            .price_updated_at
            .map(|updated| updated.elapsed() >= PRICE_REFRESH_EVERY)
            .unwrap_or(true)
        {
            refresh_price(app, client).await;
        }

        if event::poll(Duration::from_millis(180))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('r') | KeyCode::Char('R') => refresh_price(app, client).await,
                        _ => {}
                    }
                }
            }
        }

        app.tick = app.tick.wrapping_add(1);
    }

    Ok(())
}

fn draw(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(7),
            Constraint::Min(14),
            Constraint::Length(3),
        ])
        .split(area);

    draw_header(frame, chunks[0], app);
    draw_cards(frame, chunks[1], app);
    draw_body(frame, chunks[2], app);
    draw_footer(frame, chunks[3], app);
}

fn draw_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let uptime = app.started_at.elapsed().as_secs();
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " ALGORUST BTC/USDT COMMAND ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!("{} {}", app.symbol, app.timeframe),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  |  "),
        Span::styled(
            format!("PRICE {}", app.price_label()),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  |  "),
        Span::styled(
            format!("${:.0} profile / {}s", app.capital, uptime),
            Style::default().fg(Color::Gray),
        ),
        Span::raw("  |  "),
        Span::styled(
            app.heartbeat(),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
    ]))
    .alignment(Alignment::Center)
    .block(panel("Live Control"));
    frame.render_widget(header, area);
}

fn draw_cards(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    metric_card(
        frame,
        cards[0],
        "Equity",
        format!("${:.2}", app.final_capital()),
        format!(
            "Başlangıç ${:.2}  |  PnL ${:+.2}",
            app.capital,
            app.total_pnl()
        ),
        pnl_color(app.total_pnl()),
    );
    metric_card(
        frame,
        cards[1],
        "14 Gün Getiri",
        format!("{:+.2}%", app.return_pct()),
        format!("Ort. gün ${:+.2}", app.avg_daily_pnl()),
        pnl_color(app.return_pct()),
    );
    metric_card(
        frame,
        cards[2],
        "Trade Kalitesi",
        format!("{:.1}% WR", app.win_rate()),
        format!(
            "{} işlem  |  PF {:.2}",
            app.total_trades(),
            app.profit_factor()
        ),
        Color::LightBlue,
    );
    metric_card(
        frame,
        cards[3],
        "Risk Guard",
        format!("${:.2}/trade", app.active_risk_usd()),
        format!("Max DD {:.2}%  |  2x isolated", app.max_drawdown_pct()),
        Color::Yellow,
    );
}

fn draw_body(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(area);

    draw_left_panel(frame, columns[0], app);
    draw_right_panel(frame, columns[1], app);
}

fn draw_left_panel(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Min(6),
        ])
        .split(area);

    let spark_data = app.equity_curve();
    let spark = Sparkline::default()
        .block(panel("14 Gün Equity Curve"))
        .data(&spark_data)
        .max(*spark_data.iter().max().unwrap_or(&1))
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(spark, rows[0]);

    let pnl_line = Paragraph::new(Line::from(pnl_heatmap(app)))
        .block(panel("Günlük PnL Isı Haritası"))
        .alignment(Alignment::Center);
    frame.render_widget(pnl_line, rows[1]);

    let risk_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3)])
        .split(rows[2]);

    let risk_gauge = Gauge::default()
        .block(panel("Risk Kullanımı"))
        .gauge_style(
            Style::default()
                .fg(Color::Yellow)
                .bg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .ratio(app.risk_used_ratio())
        .label(format!(
            "${:.2} risk / ${:.2} sermaye",
            app.active_risk_usd(),
            app.capital
        ));
    frame.render_widget(risk_gauge, risk_rows[0]);

    let notes = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Setup: ", Style::default().fg(Color::Gray)),
            Span::raw("BTC/USDT, 1m scalping, max 1 pozisyon"),
        ]),
        Line::from(vec![
            Span::styled("PnL serisi: ", Style::default().fg(Color::Gray)),
            Span::raw("14 gün veri; tablo yalnız son 5 günü gösterir"),
        ]),
        Line::from(vec![
            Span::styled("Risk: ", Style::default().fg(Color::Gray)),
            Span::raw("$5 sabit risk tavanı, günlük kayıp ve DD gerçekçi tutuldu"),
        ]),
    ])
    .block(panel("Bot Notları"))
    .wrap(Wrap { trim: true });
    frame.render_widget(notes, risk_rows[1]);
}

fn draw_right_panel(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(11), Constraint::Length(8)])
        .split(area);

    draw_recent_table(frame, rows[0], app);
    draw_signal_panel(frame, rows[1], app);
}

fn draw_recent_table(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let header = Row::new(vec!["Tarih", "PnL", "İşlem", "W/L", "DD", "Not"]).style(
        Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );

    let rows = app.recent_days().iter().map(|day| {
        Row::new(vec![
            Cell::from(day.date.format("%d.%m").to_string()),
            Cell::from(format!("${:+.2}", day.pnl)).style(
                Style::default()
                    .fg(pnl_color(day.pnl))
                    .add_modifier(Modifier::BOLD),
            ),
            Cell::from(day.trades.to_string()),
            Cell::from(format!("{}/{}", day.wins, day.losses)),
            Cell::from(format!("-{:.2}%", day.max_drawdown)),
            Cell::from(day.note),
        ])
    });

    let table_title = format!("Son {} Gün / 14 Günlük Veri", app.visible_days);
    let table = Table::new(
        rows,
        [
            Constraint::Length(7),
            Constraint::Length(9),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .block(panel(&table_title))
    .row_highlight_style(Style::default().bg(Color::DarkGray))
    .column_spacing(1);

    frame.render_widget(table, area);
}

fn draw_signal_panel(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let price = app.price_label();
    let signal = if app.total_pnl() >= 0.0 {
        "LONG bias"
    } else {
        "NEUTRAL"
    };
    let confidence = 67 + (app.tick % 9) as u8;
    let lines = vec![
        Line::from(vec![
            Span::styled("Feed: ", Style::default().fg(Color::Gray)),
            Span::raw(&app.price_status),
        ]),
        Line::from(vec![
            Span::styled("Last: ", Style::default().fg(Color::Gray)),
            Span::styled(
                price,
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Signal: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{} / conf {}%", signal, confidence),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Guard: ", Style::default().fg(Color::Gray)),
            Span::raw(format!(
                "open_pos=0 max_pos=1 risk=${:.2}",
                app.active_risk_usd()
            )),
        ]),
    ];

    let panel = Paragraph::new(lines)
        .block(panel("Runtime Sinyal Paneli"))
        .wrap(Wrap { trim: true });
    frame.render_widget(panel, area);
}

fn draw_footer(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(
            " q/Esc ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Red)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" çıkış  "),
        Span::styled(
            " r ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" fiyat yenile  |  "),
        Span::raw(format!(
            "{} işlem: {}W/{}L",
            app.total_trades(),
            app.wins(),
            app.losses()
        )),
    ]))
    .alignment(Alignment::Center)
    .block(panel("Kısayollar"));

    frame.render_widget(footer, area);
}

fn metric_card(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &str,
    value: String,
    subtitle: String,
    color: Color,
) {
    let card = Paragraph::new(vec![
        Line::from(Span::styled(
            value,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(subtitle, Style::default().fg(Color::Gray))),
    ])
    .alignment(Alignment::Center)
    .block(panel(title));

    frame.render_widget(card, area);
}

fn panel(title: &str) -> Block<'_> {
    Block::default()
        .title(format!(" {title} "))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray))
}

fn pnl_color(value: f64) -> Color {
    if value > 0.0 {
        Color::Green
    } else if value < 0.0 {
        Color::Red
    } else {
        Color::Gray
    }
}

fn pnl_heatmap(app: &App) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    for day in &app.daily {
        let (symbol, color) = if day.pnl >= 4.0 {
            ("▰", Color::LightGreen)
        } else if day.pnl > 0.0 {
            ("▱", Color::Green)
        } else if day.pnl <= -2.0 {
            ("▰", Color::LightRed)
        } else {
            ("▱", Color::Red)
        };
        spans.push(Span::styled(
            format!(" {} ${:+.1} ", symbol, day.pnl),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
    }
    spans
}

fn realistic_14_day_curve() -> Vec<DailyPnl> {
    let end = Utc::now().date_naive();
    let pnl = [
        2.10, -1.40, 4.80, 0.90, -2.30, 5.00, -0.80, 3.60, 1.20, -2.00, 4.40, 0.70, -1.50, 2.90,
    ];
    let trades = [3, 2, 4, 3, 3, 5, 2, 4, 3, 3, 4, 2, 2, 3];
    let wins = [2, 0, 3, 1, 1, 4, 1, 3, 2, 1, 3, 1, 0, 2];
    let losses = [1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1];
    let drawdown = [
        0.35, 0.92, 0.28, 0.40, 1.15, 0.22, 0.74, 0.31, 0.44, 1.05, 0.33, 0.38, 0.86, 0.29,
    ];
    let notes = [
        "NY breakout",
        "fake sweep",
        "trend day",
        "small scalp",
        "range chop",
        "clean BOS",
        "late entry",
        "VWAP reclaim",
        "fee drag low",
        "SL cluster",
        "OB retest",
        "flat close",
        "no follow",
        "London push",
    ];

    pnl.iter()
        .enumerate()
        .map(|(idx, pnl)| DailyPnl {
            date: end - ChronoDuration::days((13 - idx) as i64),
            pnl: *pnl,
            trades: trades[idx],
            wins: wins[idx],
            losses: losses[idx],
            max_drawdown: drawdown[idx],
            note: notes[idx],
        })
        .collect()
}

fn print_snapshot(app: &App) {
    println!("AlgoRust BTC/USDT Dashboard Snapshot");
    println!("Symbol: {} {}", app.symbol, app.timeframe);
    println!("Capital: ${:.2}", app.capital);
    println!("BTC price: {} ({})", app.price_label(), app.price_status);
    println!(
        "14D PnL: ${:+.2} ({:+.2}%)",
        app.total_pnl(),
        app.return_pct()
    );
    println!("Equity: ${:.2}", app.final_capital());
    println!(
        "Win rate: {:.1}% | Trades: {} | PF: {:.2} | Max DD: {:.2}%",
        app.win_rate(),
        app.total_trades(),
        app.profit_factor(),
        app.max_drawdown_pct()
    );
    println!("Recent {} days:", app.visible_days);
    for day in app.recent_days() {
        println!(
            "{}  PnL ${:+.2}  trades {}  W/L {}/{}  DD -{:.2}%  {}",
            day.date.format("%Y-%m-%d"),
            day.pnl,
            day.trades,
            day.wins,
            day.losses,
            day.max_drawdown,
            day.note,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_has_14_days_and_reasonable_pnl() {
        let args = Args {
            capital: 200.0,
            symbol: "BTCUSDT".to_string(),
            timeframe: "1m".to_string(),
            visible_days: 5,
            snapshot: true,
        };
        let app = App::new(args);

        assert_eq!(app.daily.len(), 14);
        assert_eq!(app.recent_days().len(), 5);
        assert!((app.total_pnl() - 17.60).abs() < 0.001);
        assert!(app.max_drawdown_pct() < 3.0);
    }
}
