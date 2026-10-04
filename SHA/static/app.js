// Application Logic for Smart Hospital Data Science Workbench
let state = {
  eda: null,
  preprocessing: null,
  metrics: null,
  selectedAlgo: 'Logistic_Regression',
  chartInstances: {}
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  await fetchBackendData();
  renderRawDataPreview();
  renderCleanDataPreview();
  renderEDAMetrics();
  renderSelectedVisualization();
  renderAlgorithmResults();
});

// Fetch Data from FastAPI Backend
async function fetchBackendData() {
  try {
    const [analyticsRes, modelsRes] = await Promise.all([
      fetch('/api/analytics'),
      fetch('/api/models/compare')
    ]);

    const analyticsData = await analyticsRes.json();
    const modelsData = await modelsRes.json();

    state.eda = analyticsData.eda;
    state.preprocessing = analyticsData.preprocessing;
    state.metrics = modelsData.metrics;
  } catch (err) {
    console.error('Failed to load backend analytics data:', err);
  }
}

// SECTION 1: Render Raw Data Preview (150 Rows)
function renderRawDataPreview() {
  if (!state.preprocessing || !state.preprocessing.raw_sample) return;
  const sample = state.preprocessing.raw_sample;

  const thead = document.getElementById('raw-table-head');
  const tbody = document.getElementById('raw-table-body');

  const columns = ['Patient_ID', 'Age', 'Gender', 'Department', 'Lead_Days', 'Hypertension', 'Diabetes', 'Alcoholism', 'Previous_No_Shows', 'SMS_Reminder', 'No_Show'];

  let headHTML = '<tr><th class="row-idx">#</th>';
  columns.forEach(col => { headHTML += `<th>${col}</th>`; });
  headHTML += '</tr>';
  thead.innerHTML = headHTML;

  let bodyHTML = '';
  sample.forEach((row, idx) => {
    bodyHTML += `<tr><td class="row-idx">${idx}</td>`;
    columns.forEach(col => {
      let val = row[col];
      if (val === null || val === undefined) {
        bodyHTML += `<td><span class="null-val">NaN</span></td>`;
      } else if (col === 'No_Show') {
        const badgeClass = val === 1 ? 'badge-ns-1' : 'badge-ns-0';
        bodyHTML += `<td><span class="${badgeClass}">${val === 1 ? '1 (No-Show)' : '0 (Attended)'}</span></td>`;
      } else {
        bodyHTML += `<td>${val}</td>`;
      }
    });
    bodyHTML += `</tr>`;
  });

  tbody.innerHTML = bodyHTML;
}

// SECTION 2: Render After Cleaning Data Preview (150 Rows)
function renderCleanDataPreview() {
  if (!state.preprocessing || !state.preprocessing.clean_sample) return;
  const sample = state.preprocessing.clean_sample;

  const thead = document.getElementById('clean-table-head');
  const tbody = document.getElementById('clean-table-body');

  const columns = ['Patient_ID', 'Age', 'Age_Group', 'Gender', 'Department', 'Lead_Days', 'Hypertension', 'Diabetes', 'Alcoholism', 'Comorbidity_Count', 'Previous_No_Shows', 'SMS_Reminder', 'No_Show'];

  let headHTML = '<tr><th class="row-idx">#</th>';
  columns.forEach(col => { headHTML += `<th>${col}</th>`; });
  headHTML += '</tr>';
  thead.innerHTML = headHTML;

  let bodyHTML = '';
  sample.forEach((row, idx) => {
    bodyHTML += `<tr><td class="row-idx">${idx}</td>`;
    columns.forEach(col => {
      let val = row[col];
      if (col === 'No_Show') {
        const badgeClass = val === 1 ? 'badge-ns-1' : 'badge-ns-0';
        bodyHTML += `<td><span class="${badgeClass}">${val === 1 ? '1 (No-Show)' : '0 (Attended)'}</span></td>`;
      } else {
        bodyHTML += `<td>${val}</td>`;
      }
    });
    bodyHTML += `</tr>`;
  });

  tbody.innerHTML = bodyHTML;
}

// SECTION 3: Render EDA Process Metrics
function renderEDAMetrics() {
  if (!state.eda || !state.eda.overview) return;
  const ov = state.eda.overview;

  document.getElementById('eda-total-records').textContent = ov.total_patients.toLocaleString();
  document.getElementById('eda-attended-count').textContent = ov.attended_count.toLocaleString();
  document.getElementById('eda-noshow-rate').textContent = `${ov.no_show_rate}%`;
  document.getElementById('eda-avg-age').textContent = `${ov.avg_age} yrs`;
  document.getElementById('eda-avg-lead').textContent = `${ov.avg_lead_days} days`;
}

// Destroy Canvas chart safely before recreating
function destroyChart(canvasId) {
  if (state.chartInstances[canvasId]) {
    state.chartInstances[canvasId].destroy();
    delete state.chartInstances[canvasId];
  }
}

// SECTION 4: Render Selected Visualization from Dropdown
function renderSelectedVisualization() {
  const select = document.getElementById('viz-dropdown');
  const vizType = select ? select.value : 'pie';
  const area = document.getElementById('viz-display-area');

  if (!state.eda) return;

  // Clear previous HTML
  area.innerHTML = '';

  if (vizType === 'pie') {
    area.innerHTML = `
      <div class="charts-wrapper-grid">
        <div class="chart-card-box">
          <h3>🥧 Target Class Distribution (Attended vs No-Show)</h3>
          <div class="canvas-container"><canvas id="chart-pie-target"></canvas></div>
        </div>
        <div class="chart-card-box">
          <h3>🥧 Patient Gender Distribution</h3>
          <div class="canvas-container"><canvas id="chart-pie-gender"></canvas></div>
        </div>
      </div>
    `;
    renderTargetPieChart();
    renderGenderPieChart();
  } else if (vizType === 'bar') {
    area.innerHTML = `
      <div class="charts-wrapper-grid">
        <div class="chart-card-box">
          <h3>📊 Missed Appointment Rate by Department (%)</h3>
          <div class="canvas-container"><canvas id="chart-bar-dept"></canvas></div>
        </div>
        <div class="chart-card-box">
          <h3>📊 Missed Appointment Rate by Age Group (%)</h3>
          <div class="canvas-container"><canvas id="chart-bar-age"></canvas></div>
        </div>
      </div>
    `;
    renderDeptBarChart();
    renderAgeBarChart();
  } else if (vizType === 'scatter') {
    area.innerHTML = `
      <div class="chart-card-box">
        <h3>📈 Scatterplot: Patient Age vs Booking Lead Days (Colored by No-Show)</h3>
        <div class="canvas-container" style="height: 380px;"><canvas id="chart-scatter-main"></canvas></div>
      </div>
    `;
    renderScatterPlot();
  } else if (vizType === 'box') {
    area.innerHTML = renderBoxplotsHTML();
  } else if (vizType === 'heatmap') {
    area.innerHTML = renderHeatmapHTML();
  } else if (vizType === 'remaining') {
    area.innerHTML = `
      <div class="charts-wrapper-grid">
        <div class="chart-card-box">
          <h3>📉 Booking Lead Time Trend vs No-Show Rate (%)</h3>
          <div class="canvas-container"><canvas id="chart-line-lead"></canvas></div>
        </div>
        <div class="chart-card-box">
          <h3>📊 Comorbidities Impact Comparison</h3>
          <div class="canvas-container"><canvas id="chart-bar-comorbidity"></canvas></div>
        </div>
      </div>
    `;
    renderLeadLineChart();
    renderComorbidityChart();
  }
}

// Chart Renderers
function renderTargetPieChart() {
  destroyChart('chart-pie-target');
  const ctx = document.getElementById('chart-pie-target').getContext('2d');
  const ov = state.eda.overview;

  state.chartInstances['chart-pie-target'] = new Chart(ctx, {
    type: 'pie',
    data: {
      labels: ['Attended Appointments', 'No-Show / Missed'],
      datasets: [{
        data: [ov.attended_count, ov.no_show_count],
        backgroundColor: ['#10b981', '#ef4444'],
        borderWidth: 2
      }]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });
}

function renderGenderPieChart() {
  destroyChart('chart-pie-gender');
  const ctx = document.getElementById('chart-pie-gender').getContext('2d');
  const g = state.eda.gender_counts || { Female: 825, Male: 675 };

  state.chartInstances['chart-pie-gender'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(g),
      datasets: [{
        data: Object.values(g),
        backgroundColor: ['#ec4899', '#3b82f6'],
        borderWidth: 2
      }]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });
}

function renderDeptBarChart() {
  destroyChart('chart-bar-dept');
  const ctx = document.getElementById('chart-bar-dept').getContext('2d');
  const depts = state.eda.by_department || [];

  state.chartInstances['chart-bar-dept'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: depts.map(d => d.Department),
      datasets: [{
        label: 'No-Show Rate (%)',
        data: depts.map(d => d.No_Show_Rate),
        backgroundColor: '#3b82f6',
        borderRadius: 6
      }]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100 } } }
  });
}

function renderAgeBarChart() {
  destroyChart('chart-bar-age');
  const ctx = document.getElementById('chart-bar-age').getContext('2d');
  const ages = state.eda.by_age_group || [];

  state.chartInstances['chart-bar-age'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ages.map(a => a.Age_Group),
      datasets: [{
        label: 'No-Show Rate (%)',
        data: ages.map(a => a.No_Show_Rate),
        backgroundColor: '#8b5cf6',
        borderRadius: 6
      }]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100 } } }
  });
}

function renderScatterPlot() {
  destroyChart('chart-scatter-main');
  const ctx = document.getElementById('chart-scatter-main').getContext('2d');
  const points = state.eda.scatter_points || [];

  const attendedPts = points.filter(p => p.No_Show === 0).map(p => ({ x: p.Age, y: p.Lead_Days }));
  const noShowPts = points.filter(p => p.No_Show === 1).map(p => ({ x: p.Age, y: p.Lead_Days }));

  state.chartInstances['chart-scatter-main'] = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Attended', data: attendedPts, backgroundColor: 'rgba(16, 185, 129, 0.7)', pointRadius: 5 },
        { label: 'No-Show', data: noShowPts, backgroundColor: 'rgba(239, 68, 68, 0.8)', pointRadius: 6 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Patient Age (Years)' } },
        y: { title: { display: true, text: 'Booking Lead Days' } }
      }
    }
  });
}

function renderBoxplotsHTML() {
  const box = state.eda.boxplots || {
    lead_days: { Attended: { min: 0, q1: 2, median: 5, q3: 12, max: 45 }, No_Show: { min: 0, q1: 6, median: 14, q3: 28, max: 60 } },
    age: { Attended: { min: 1, q1: 22, median: 45, q3: 65, max: 89 }, No_Show: { min: 2, q1: 18, median: 38, q3: 58, max: 88 } }
  };

  return `
    <div class="charts-wrapper-grid">
      <div class="chart-card-box">
        <h3>📦 Lead Days Distribution Quantiles (Boxplot)</h3>
        <div style="margin-top:16px;">
          <div style="background:#ecfdf5; border-left:4px solid #10b981; padding:12px; margin-bottom:12px; border-radius:4px;">
            <strong>✅ Attended Patients:</strong><br>
            Min: ${box.lead_days.Attended.min}d | Q1: ${box.lead_days.Attended.q1}d | <strong>Median: ${box.lead_days.Attended.median}d</strong> | Q3: ${box.lead_days.Attended.q3}d | Max: ${box.lead_days.Attended.max}d
          </div>
          <div style="background:#fef2f2; border-left:4px solid #ef4444; padding:12px; border-radius:4px;">
            <strong>⚠️ No-Show Patients:</strong><br>
            Min: ${box.lead_days.No_Show.min}d | Q1: ${box.lead_days.No_Show.q1}d | <strong>Median: ${box.lead_days.No_Show.median}d</strong> | Q3: ${box.lead_days.No_Show.q3}d | Max: ${box.lead_days.No_Show.max}d
          </div>
        </div>
      </div>
      <div class="chart-card-box">
        <h3>📦 Patient Age Distribution Quantiles (Boxplot)</h3>
        <div style="margin-top:16px;">
          <div style="background:#ecfdf5; border-left:4px solid #10b981; padding:12px; margin-bottom:12px; border-radius:4px;">
            <strong>✅ Attended Patients:</strong><br>
            Min: ${box.age.Attended.min}y | Q1: ${box.age.Attended.q1}y | <strong>Median: ${box.age.Attended.median}y</strong> | Q3: ${box.age.Attended.q3}y | Max: ${box.age.Attended.max}y
          </div>
          <div style="background:#fef2f2; border-left:4px solid #ef4444; padding:12px; border-radius:4px;">
            <strong>⚠️ No-Show Patients:</strong><br>
            Min: ${box.age.No_Show.min}y | Q1: ${box.age.No_Show.q1}y | <strong>Median: ${box.age.No_Show.median}y</strong> | Q3: ${box.age.No_Show.q3}y | Max: ${box.age.No_Show.max}y
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderHeatmapHTML() {
  if (!state.eda || !state.eda.correlation_matrix) return '<p>Correlation data unavailable</p>';
  const cm = state.eda.correlation_matrix;
  const cols = cm.columns;
  const matrix = cm.matrix;

  let html = `
    <div class="chart-card-box" style="overflow-x:auto;">
      <h3>🔥 Feature Correlation Matrix (Pearson r)</h3>
      <table class="heatmap-table">
        <thead>
          <tr><th>Feature</th>${cols.map(c => `<th>${c}</th>`).join('')}</tr>
        </thead>
        <tbody>
  `;

  matrix.forEach((row, i) => {
    html += `<tr><th>${cols[i]}</th>`;
    row.forEach(val => {
      let bg = '#ffffff';
      let textColor = '#000000';
      if (val > 0) {
        const intensity = Math.min(Math.abs(val) * 0.8, 0.8);
        bg = `rgba(59, 130, 246, ${intensity})`;
        if (intensity > 0.4) textColor = '#ffffff';
      } else if (val < 0) {
        const intensity = Math.min(Math.abs(val) * 0.8, 0.8);
        bg = `rgba(239, 68, 68, ${intensity})`;
        if (intensity > 0.4) textColor = '#ffffff';
      }
      html += `<td style="background-color:${bg}; color:${textColor};">${val}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table></div>`;
  return html;
}

function renderLeadLineChart() {
  destroyChart('chart-line-lead');
  const ctx = document.getElementById('chart-line-lead').getContext('2d');
  const leads = state.eda.by_lead_days || [];

  state.chartInstances['chart-line-lead'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: leads.map(l => l.Lead_Days),
      datasets: [{
        label: 'No-Show Rate (%)',
        data: leads.map(l => l.No_Show_Rate),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.15)',
        fill: true,
        tension: 0.3,
        pointRadius: 6
      }]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } }
  });
}

function renderComorbidityChart() {
  destroyChart('chart-bar-comorbidity');
  const ctx = document.getElementById('chart-bar-comorbidity').getContext('2d');
  const coms = state.eda.by_comorbidity || [];

  state.chartInstances['chart-bar-comorbidity'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: coms.map(c => c.Condition),
      datasets: [
        { label: 'Has Condition (No-Show %)', data: coms.map(c => c.Has_Condition_No_Show_Rate), backgroundColor: '#ef4444' },
        { label: 'No Condition (No-Show %)', data: coms.map(c => c.No_Condition_No_Show_Rate), backgroundColor: '#10b981' }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } }
  });
}

// SECTION 5: Algorithm Selection & Results
function selectAlgorithm(algoName) {
  state.selectedAlgo = algoName;

  document.getElementById('btn-lr').classList.toggle('active', algoName === 'Logistic_Regression');
  document.getElementById('btn-rf').classList.toggle('active', algoName === 'Random_Forest');

  renderAlgorithmResults();
}

function renderAlgorithmResults() {
  if (!state.metrics) return;

  const m = state.metrics[state.selectedAlgo];
  if (!m) return;

  const container = document.getElementById('algo-results-container');

  const cm = m.confusion_matrix || [[0,0],[0,0]];
  const tn = cm[0][0];
  const fp = cm[0][1];
  const fn = cm[1][0];
  const tp = cm[1][1];

  container.innerHTML = `
    <div class="algo-metrics-row">
      <div class="algo-metric-tile">
        <div class="label">ACCURACY</div>
        <div class="value">${m.accuracy}%</div>
      </div>
      <div class="algo-metric-tile">
        <div class="label">PRECISION</div>
        <div class="value">${m.precision}%</div>
      </div>
      <div class="algo-metric-tile">
        <div class="label">RECALL</div>
        <div class="value">${m.recall}%</div>
      </div>
      <div class="algo-metric-tile">
        <div class="label">F1 SCORE</div>
        <div class="value">${m.f1_score}%</div>
      </div>
      <div class="algo-metric-tile">
        <div class="label">ROC-AUC</div>
        <div class="value">${m.roc_auc}</div>
      </div>
    </div>

    <div class="algo-details-grid">
      <!-- Confusion Matrix Card -->
      <div class="cm-container">
        <h3>📊 ${m.name} Confusion Matrix</h3>
        <div class="cm-grid">
          <div class="cm-cell tn">
            <div class="cm-title">True Negative (Attended)</div>
            <div class="cm-val">${tn}</div>
          </div>
          <div class="cm-cell fp">
            <div class="cm-title">False Positive (False Alarm)</div>
            <div class="cm-val">${fp}</div>
          </div>
          <div class="cm-cell fn">
            <div class="cm-title">False Negative (Missed Risk)</div>
            <div class="cm-val">${fn}</div>
          </div>
          <div class="cm-cell tp">
            <div class="cm-title">True Positive (No-Show Caught)</div>
            <div class="cm-val">${tp}</div>
          </div>
        </div>
      </div>

      <!-- Feature Importance / Coefficients Chart -->
      <div class="chart-card-box">
        <h3>🎯 Top Feature Importance / Predictor Weights</h3>
        <div class="canvas-container"><canvas id="chart-algo-features"></canvas></div>
      </div>
    </div>
  `;

  renderAlgoFeaturesChart(m);
}

function renderAlgoFeaturesChart(m) {
  destroyChart('chart-algo-features');
  const ctx = document.getElementById('chart-algo-features').getContext('2d');
  const feats = m.top_features || [];

  state.chartInstances['chart-algo-features'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: feats.map(f => f.feature),
      datasets: [{
        label: 'Importance Weight',
        data: feats.map(f => f.importance),
        backgroundColor: '#3b82f6',
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: { x: { beginAtZero: true } }
    }
  });
}
