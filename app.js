"use strict";

/*
 * PulseBoard front-end.
 *
 * Loads data/gapminder.json (raw country-year rows) and data/findings.json
 * (precomputed Python results) and renders:
 *   - hero headline stats (from findings.json, server-computed)
 *   - a log-x scatter plot of gdpPercap vs lifeExp with an OLS fit line
 *   - a bar chart of mean lifeExp per continent with 95% CI error bars
 *   - continent + year-range filters that recompute the OLS fit and the
 *     per-continent means/CIs CLIENT-SIDE, on the filtered subset, via the
 *     closed-form normal equations + a t-distribution based confidence
 *     interval/p-value (implemented below, no library).
 *
 * No eval(). No innerHTML of dynamic/data-derived content — all DOM nodes
 * for chart geometry and text are built with createElementNS/createElement
 * and set via textContent / attribute setters.
 */

const CONTINENT_COLORS = {
  Africa: "#ff7a59",
  Americas: "#00e0c7",
  Asia: "#c6f135",
  Europe: "#7aa2ff",
  Oceania: "#e879f9",
};
const CONTINENT_ORDER = ["Africa", "Americas", "Asia", "Europe", "Oceania"];

let RAW_DATA = [];
let FINDINGS = null;

const state = {
  continents: new Set(CONTINENT_ORDER),
  yearMin: 1952,
  yearMax: 2007,
};

// ---------------------------------------------------------------------------
// Statistics: t-distribution helpers (no library — small, self-contained
// implementation good enough for CI / p-value display purposes).
// ---------------------------------------------------------------------------

/** Regularized incomplete beta function via continued fraction (Numerical Recipes). */
function betacf(x, a, b) {
  const MAXIT = 200;
  const EPS = 3e-9;
  const FPMIN = 1e-30;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function logGamma(x) {
  const cof = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  const g = 7;
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + g + 0.5;
  for (let i = 0; i < cof.length; i++) {
    a += cof[i] / (x + i + 1);
  }
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function incompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(
    logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x)
  );
  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betacf(x, a, b)) / a;
  }
  return 1 - (bt * betacf(1 - x, b, a)) / b;
}

/** Two-sided p-value for Student's t distribution with `df` degrees of freedom. */
function tTwoSidedPValue(t, df) {
  const x = df / (df + t * t);
  return incompleteBeta(x, df / 2, 0.5);
}

/** Inverse CDF (quantile) of Student's t via bisection on the CDF — robust, no recursion blowup. */
function tCriticalValue(df, alpha) {
  // two-sided critical value such that P(|T| > t*) = alpha
  let lo = 0;
  let hi = 1000;
  const target = alpha;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    const p = tTwoSidedPValue(mid, df);
    if (p > target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo + hi) / 2;
}

/**
 * Simple linear regression y = a + b*x via closed-form normal equations,
 * with standard error, two-sided p-value, and 95% CI for the slope.
 */
function simpleOLS(xs, ys) {
  const n = xs.length;
  if (n < 3) {
    return { slope: NaN, intercept: NaN, se: NaN, ci95Low: NaN, ci95High: NaN, p: NaN, r2: NaN, n };
  }
  let sumX = 0, sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i];
    sumY += ys[i];
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }

  const slope = sxy / sxx;
  const intercept = meanY - slope * meanX;

  // residual sum of squares
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const pred = intercept + slope * xs[i];
    const resid = ys[i] - pred;
    rss += resid * resid;
  }
  const df = n - 2;
  const mse = rss / df;
  const seSlope = Math.sqrt(mse / sxx);
  const tStat = slope / seSlope;
  const p = tTwoSidedPValue(tStat, df);
  const tCrit = tCriticalValue(df, 0.05);
  const ci95Low = slope - tCrit * seSlope;
  const ci95High = slope + tCrit * seSlope;
  const r2 = 1 - rss / syy;

  return { slope, intercept, se: seSlope, ci95Low, ci95High, p, r2, n, df };
}

function meanCI95(values) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  if (n < 2) {
    return { mean, ci95Low: mean, ci95High: mean, n };
  }
  let ss = 0;
  for (const v of values) ss += (v - mean) * (v - mean);
  const variance = ss / (n - 1);
  const sem = Math.sqrt(variance / n);
  const tCrit = tCriticalValue(n - 1, 0.05);
  const h = tCrit * sem;
  return { mean, ci95Low: mean - h, ci95High: mean + h, n };
}

function oneWayAnovaF(groups) {
  // groups: array of arrays of numbers
  const all = groups.flat();
  const grandMean = all.reduce((a, b) => a + b, 0) / all.length;
  let ssBetween = 0;
  let ssWithin = 0;
  let dfBetween = groups.length - 1;
  let dfWithin = 0;
  for (const g of groups) {
    const gMean = g.reduce((a, b) => a + b, 0) / g.length;
    ssBetween += g.length * (gMean - grandMean) * (gMean - grandMean);
    for (const v of g) ssWithin += (v - gMean) * (v - gMean);
    dfWithin += g.length - 1;
  }
  const msBetween = ssBetween / dfBetween;
  const msWithin = ssWithin / dfWithin;
  const f = msBetween / msWithin;
  return { f, dfBetween, dfWithin };
}

/** Approximate two-sided p-value for the F distribution via the incomplete beta relation. */
function fPValue(f, df1, df2) {
  if (!isFinite(f) || f <= 0) return 1;
  const x = df2 / (df2 + df1 * f);
  return incompleteBeta(x, df2 / 2, df1 / 2);
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function fmtNum(x, digits = 2) {
  if (!isFinite(x)) return "—";
  return x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtP(p) {
  if (!isFinite(p)) return "—";
  if (p < 0.0001) return "p < 0.0001";
  return `p = ${p.toFixed(4)}`;
}

function fmtPShort(p) {
  if (!isFinite(p)) return "—";
  if (p < 0.0001) return "< 0.0001";
  return p.toFixed(4);
}

// ---------------------------------------------------------------------------
// SVG helpers (no innerHTML — all nodes built via DOM APIs)
// ---------------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  return el;
}

function clearSvg(svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  // re-add title/desc which are required for accessibility and were wiped
}

function textEl(tag_attrs, content, className) {
  const el = svgEl("text", tag_attrs);
  if (className) el.setAttribute("class", className);
  el.textContent = content; // safe: textContent, never innerHTML
  return el;
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

async function loadData() {
  const [rawResp, findingsResp] = await Promise.all([
    fetch("data/gapminder.json"),
    fetch("data/findings.json"),
  ]);
  if (!rawResp.ok) throw new Error("Failed to load gapminder.json");
  if (!findingsResp.ok) throw new Error("Failed to load findings.json");
  RAW_DATA = await rawResp.json();
  FINDINGS = await findingsResp.json();
}

function getFilteredRows() {
  return RAW_DATA.filter(
    (d) =>
      state.continents.has(d.continent) &&
      d.year >= state.yearMin &&
      d.year <= state.yearMax
  );
}

// ---------------------------------------------------------------------------
// Rendering: hero stats (server-computed, full dataset)
// ---------------------------------------------------------------------------

function renderHero() {
  const h = FINDINGS.headline;
  document.getElementById("hero-n").textContent = h.n.toLocaleString("en-US");
  document.getElementById("stat-slope").textContent = `+${fmtNum(h.slope, 2)} yrs`;
  document.getElementById("stat-slope-ci").textContent =
    `95% CI [${fmtNum(h.ci95_low, 2)}, ${fmtNum(h.ci95_high, 2)}]`;
  document.getElementById("stat-p").textContent = h.p_value < 1e-15 ? "p < 1e-15" : fmtP(h.p_value);
  document.getElementById("stat-r2").textContent = fmtNum(h.r_squared, 3);
  document.getElementById("stat-n").textContent = h.n.toLocaleString("en-US");

  const ds = FINDINGS.dataset;
  const methodEl = document.getElementById("methodology-dataset");
  methodEl.textContent = "";
  const p1 = document.createElement("span");
  p1.textContent = `Source: `;
  const link = document.createElement("a");
  link.href = ds.source_url;
  link.rel = "noopener";
  link.textContent = ds.source_url;
  methodEl.appendChild(p1);
  methodEl.appendChild(link);
  const rest = document.createElement("span");
  rest.textContent = ` — downloaded ${ds.download_date}, SHA-256 ${ds.sha256}. ${ds.license_note} n = ${FINDINGS.meta.n_rows.toLocaleString("en-US")} country-year rows across ${FINDINGS.meta.n_countries} countries, ${FINDINGS.meta.year_min}–${FINDINGS.meta.year_max}.`;
  methodEl.appendChild(rest);
}

// ---------------------------------------------------------------------------
// Rendering: filter controls
// ---------------------------------------------------------------------------

function renderFilterChips() {
  const row = document.getElementById("continent-chips");
  row.textContent = "";
  for (const continent of CONTINENT_ORDER) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = continent;
    chip.setAttribute("aria-pressed", state.continents.has(continent) ? "true" : "false");
    if (state.continents.has(continent)) {
      chip.style.background = CONTINENT_COLORS[continent];
      chip.style.borderColor = CONTINENT_COLORS[continent];
    }
    chip.addEventListener("click", () => {
      if (state.continents.has(continent)) {
        if (state.continents.size === 1) return; // keep at least one selected
        state.continents.delete(continent);
      } else {
        state.continents.add(continent);
      }
      renderFilterChips();
      update();
    });
    row.appendChild(chip);
  }
}

function renderYearControls() {
  const minSlider = document.getElementById("year-min");
  const maxSlider = document.getElementById("year-max");
  const readout = document.getElementById("year-readout");

  const sync = () => {
    let lo = parseInt(minSlider.value, 10);
    let hi = parseInt(maxSlider.value, 10);
    if (lo > hi) {
      [lo, hi] = [hi, lo];
    }
    state.yearMin = lo;
    state.yearMax = hi;
    readout.textContent = `${lo} – ${hi}`;
    update();
  };

  minSlider.addEventListener("input", sync);
  maxSlider.addEventListener("input", sync);

  document.getElementById("reset-filters").addEventListener("click", () => {
    state.continents = new Set(CONTINENT_ORDER);
    state.yearMin = 1952;
    state.yearMax = 2007;
    minSlider.value = "1952";
    maxSlider.value = "2007";
    readout.textContent = "1952 – 2007";
    renderFilterChips();
    update();
  });
}

// ---------------------------------------------------------------------------
// Rendering: scatter chart
// ---------------------------------------------------------------------------

const SCATTER = { width: 920, height: 520, margin: { top: 24, right: 24, bottom: 56, left: 64 } };

function renderScatter(rows, fit) {
  const svg = document.getElementById("scatter-chart");
  clearSvg(svg);
  // restore title/desc (cleared above) for a11y
  svg.appendChild((() => { const t = svgEl("title"); t.textContent = "Scatter plot of GDP per capita versus life expectancy"; return t; })());
  svg.appendChild((() => { const d = svgEl("desc"); d.textContent = "Log-scaled GDP per capita on the x-axis, life expectancy in years on the y-axis, points colored by continent, with an overlaid OLS regression line."; return d; })());

  const { width, height, margin } = SCATTER;
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const gdpVals = rows.map((d) => d.gdpPercap);
  const logMin = Math.log10(Math.min(...gdpVals) * 0.9);
  const logMax = Math.log10(Math.max(...gdpVals) * 1.05);
  const yMin = 15;
  const yMax = 90;

  const xScale = (gdp) => {
    const l = Math.log10(gdp);
    return margin.left + ((l - logMin) / (logMax - logMin)) * plotW;
  };
  const yScale = (life) => margin.top + plotH - ((life - yMin) / (yMax - yMin)) * plotH;

  // grid + axes
  const gridGroup = svgEl("g");
  const xTicks = [100, 1000, 10000, 100000];
  for (const t of xTicks) {
    if (Math.log10(t) < logMin || Math.log10(t) > logMax) continue;
    const x = xScale(t);
    gridGroup.appendChild(svgEl("line", { x1: x, x2: x, y1: margin.top, y2: margin.top + plotH, class: "grid-line" }));
    gridGroup.appendChild(textEl({ x, y: height - margin.bottom + 20, "text-anchor": "middle" }, "$" + t.toLocaleString("en-US"), "axis-label"));
  }
  const yTicks = [20, 35, 50, 65, 80];
  for (const t of yTicks) {
    const y = yScale(t);
    gridGroup.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "grid-line" }));
    gridGroup.appendChild(textEl({ x: margin.left - 10, y: y + 4, "text-anchor": "end" }, String(t), "axis-label"));
  }
  svg.appendChild(gridGroup);

  svg.appendChild(textEl({ x: width / 2, y: height - 12, "text-anchor": "middle" }, "GDP per capita (log scale, USD)", "axis-title"));
  svg.appendChild(
    svgEl("text", {
      x: 18,
      y: margin.top + plotH / 2,
      "text-anchor": "middle",
      transform: `rotate(-90 18 ${margin.top + plotH / 2})`,
      class: "axis-title",
    })
  );
  const yTitle = svg.lastChild;
  yTitle.textContent = "Life expectancy (years)";

  // points
  const ptsGroup = svgEl("g");
  for (const d of rows) {
    const c = svgEl("circle", {
      cx: xScale(d.gdpPercap),
      cy: yScale(d.lifeExp),
      r: 3.2,
      fill: CONTINENT_COLORS[d.continent] || "#888",
      "fill-opacity": 0.55,
      stroke: "none",
    });
    const t = svgEl("title");
    t.textContent = `${d.country}, ${d.year}: $${Math.round(d.gdpPercap).toLocaleString("en-US")}, ${d.lifeExp.toFixed(1)}y`;
    c.appendChild(t);
    ptsGroup.appendChild(c);
  }
  svg.appendChild(ptsGroup);

  // regression line across the visible x-range
  if (fit && isFinite(fit.slope)) {
    const xStartLog = logMin;
    const xEndLog = logMax;
    const x1 = margin.left;
    const x2 = margin.left + plotW;
    const y1 = yScale(fit.intercept + fit.slope * xStartLog);
    const y2 = yScale(fit.intercept + fit.slope * xEndLog);
    svg.appendChild(svgEl("line", { x1, x2, y1, y2, class: "reg-line" }));
  }

  // legend
  const legend = document.getElementById("scatter-legend");
  legend.textContent = "";
  for (const continent of CONTINENT_ORDER) {
    const item = document.createElement("span");
    item.className = "legend-item";
    const dot = document.createElement("span");
    dot.className = "legend-dot";
    dot.style.background = CONTINENT_COLORS[continent];
    item.appendChild(dot);
    const label = document.createElement("span");
    label.textContent = continent;
    item.appendChild(label);
    legend.appendChild(item);
  }
  const lineItem = document.createElement("span");
  lineItem.className = "legend-item";
  const lineSwatch = document.createElement("span");
  lineSwatch.style.display = "inline-block";
  lineSwatch.style.width = "14px";
  lineSwatch.style.height = "2.5px";
  lineSwatch.style.background = "#c6f135";
  lineItem.appendChild(lineSwatch);
  const lineLabel = document.createElement("span");
  lineLabel.textContent = "OLS fit (current selection)";
  lineItem.appendChild(lineLabel);
  legend.appendChild(lineItem);
}

// ---------------------------------------------------------------------------
// Rendering: bar chart (mean lifeExp by continent, with 95% CI)
// ---------------------------------------------------------------------------

const BAR = { width: 920, height: 420, margin: { top: 24, right: 24, bottom: 56, left: 64 } };

function renderBarChart(rows) {
  const svg = document.getElementById("bar-chart");
  clearSvg(svg);
  svg.appendChild((() => { const t = svgEl("title"); t.textContent = "Bar chart of mean life expectancy by continent"; return t; })());
  svg.appendChild((() => { const d = svgEl("desc"); d.textContent = "Bars show mean life expectancy per continent for the current filter selection with 95 percent confidence interval error bars."; return d; })());

  const { width, height, margin } = BAR;
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const present = CONTINENT_ORDER.filter((c) => state.continents.has(c));
  const groupedByContinent = {};
  for (const c of present) groupedByContinent[c] = [];
  for (const d of rows) {
    if (groupedByContinent[d.continent]) groupedByContinent[d.continent].push(d.lifeExp);
  }

  const stats = present.map((c) => ({ continent: c, ...meanCI95(groupedByContinent[c].length ? groupedByContinent[c] : [0]) }));

  const yMin = 0;
  const yMax = 90;
  const yScale = (v) => margin.top + plotH - (v / yMax) * plotH;

  const bandWidth = plotW / present.length;
  const barWidth = Math.min(70, bandWidth * 0.5);

  const gridGroup = svgEl("g");
  const yTicks = [0, 20, 40, 60, 80];
  for (const t of yTicks) {
    const y = yScale(t);
    gridGroup.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "grid-line" }));
    gridGroup.appendChild(textEl({ x: margin.left - 10, y: y + 4, "text-anchor": "end" }, String(t), "axis-label"));
  }
  svg.appendChild(gridGroup);
  svg.appendChild(textEl({ x: 18, y: margin.top + plotH / 2, "text-anchor": "middle", transform: `rotate(-90 18 ${margin.top + plotH / 2})` }, "Mean life expectancy (years)", "axis-title"));

  stats.forEach((s, i) => {
    const cx = margin.left + bandWidth * i + bandWidth / 2;
    const barH = plotH - (yScale(s.mean) - margin.top);
    const barY = yScale(s.mean);
    const color = CONTINENT_COLORS[s.continent];

    svg.appendChild(
      svgEl("rect", {
        x: cx - barWidth / 2,
        y: barY,
        width: barWidth,
        height: Math.max(barH, 0),
        fill: color,
        "fill-opacity": 0.75,
        rx: 4,
      })
    );

    // error bar
    const yHi = yScale(s.ci95High);
    const yLo = yScale(s.ci95Low);
    svg.appendChild(svgEl("line", { x1: cx, x2: cx, y1: yHi, y2: yLo, stroke: "#e7edf3", "stroke-width": 1.6 }));
    svg.appendChild(svgEl("line", { x1: cx - 8, x2: cx + 8, y1: yHi, y2: yHi, stroke: "#e7edf3", "stroke-width": 1.6 }));
    svg.appendChild(svgEl("line", { x1: cx - 8, x2: cx + 8, y1: yLo, y2: yLo, stroke: "#e7edf3", "stroke-width": 1.6 }));

    svg.appendChild(textEl({ x: cx, y: height - margin.bottom + 20, "text-anchor": "middle" }, s.continent, "axis-label"));
    svg.appendChild(textEl({ x: cx, y: barY - 10, "text-anchor": "middle" }, s.mean.toFixed(1), "axis-label"));
  });
}

// ---------------------------------------------------------------------------
// Main update cycle
// ---------------------------------------------------------------------------

function update() {
  const rows = getFilteredRows();

  // client-side OLS on filtered subset
  const xs = rows.map((d) => Math.log10(d.gdpPercap));
  const ys = rows.map((d) => d.lifeExp);
  const fit = simpleOLS(xs, ys);

  document.getElementById("live-slope").textContent = isFinite(fit.slope) ? `+${fmtNum(fit.slope, 2)}` : "—";
  document.getElementById("live-ci").textContent = isFinite(fit.ci95Low)
    ? `[${fmtNum(fit.ci95Low, 2)}, ${fmtNum(fit.ci95High, 2)}]`
    : "[—, —]";
  document.getElementById("live-p").textContent = isFinite(fit.p) ? fmtPShort(fit.p) : "—";
  document.getElementById("live-r2").textContent = isFinite(fit.r2) ? fmtNum(fit.r2, 3) : "—";
  document.getElementById("live-n").textContent = rows.length.toLocaleString("en-US");

  renderScatter(rows, fit);
  renderBarChart(rows);

  // ANOVA on filtered subset (only meaningful with 2+ continents present)
  const present = CONTINENT_ORDER.filter((c) => state.continents.has(c));
  const groups = present
    .map((c) => rows.filter((d) => d.continent === c).map((d) => d.lifeExp))
    .filter((g) => g.length > 1);
  const anovaF = document.getElementById("anova-f");
  const anovaP = document.getElementById("anova-p");
  if (groups.length >= 2) {
    const { f, dfBetween, dfWithin } = oneWayAnovaF(groups);
    const p = fPValue(f, dfBetween, dfWithin);
    anovaF.textContent = fmtNum(f, 2);
    anovaP.textContent = fmtPShort(p);
  } else {
    anovaF.textContent = "—";
    anovaP.textContent = "n/a (need ≥2 continents)";
  }
}

// ---------------------------------------------------------------------------
// Self-check: client OLS vs. Python headline slope on the FULL dataset.
// Logged to console only — does not alter the UI.
// ---------------------------------------------------------------------------

function selfCheck() {
  const xs = RAW_DATA.map((d) => Math.log10(d.gdpPercap));
  const ys = RAW_DATA.map((d) => d.lifeExp);
  const fit = simpleOLS(xs, ys);
  const pySlope = FINDINGS.headline.slope;
  const diff = Math.abs(fit.slope - pySlope);
  // eslint-disable-next-line no-console
  console.log(
    `[PulseBoard self-check] JS slope=${fit.slope.toFixed(6)} vs Python slope=${pySlope.toFixed(6)} ` +
      `(abs diff=${diff.toExponential(3)})`
  );
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  try {
    await loadData();
  } catch (err) {
    document.getElementById("hero-n").textContent = "error";
    // eslint-disable-next-line no-console
    console.error("PulseBoard failed to load data:", err);
    return;
  }
  renderHero();
  renderFilterChips();
  renderYearControls();
  update();
  selfCheck();
}

document.addEventListener("DOMContentLoaded", init);
