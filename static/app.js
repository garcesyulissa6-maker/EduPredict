/* EduPredict - App JS */
const NCOLS = 36;
const CLASSES = ['Dropout','Enrolled','Graduate'];
const CLASS_COLORS = {'Dropout':'var(--red)','Enrolled':'var(--amber)','Graduate':'var(--green)'};
const BADGE_CLS = {'Dropout':'b-drop','Enrolled':'b-enr','Graduate':'b-grad'};
let modI='lr', modB='lr';

function switchTab(t){
  document.querySelectorAll('.tab').forEach((el,i)=>
    el.classList.toggle('active',(t==='ind'&&i===0)||(t==='batch'&&i===1)));
  ['ind','batch'].forEach(id=>
    document.getElementById('panel-'+id).classList.toggle('active',id===t));
}

function selMod(m,panel){
  if(panel==='i'){
    modI=m;
    document.getElementById('mi-lr').classList.toggle('sel',m==='lr');
    document.getElementById('mi-ann').classList.toggle('sel',m==='ann');
  }else{
    modB=m;
    document.getElementById('mb-lr').classList.toggle('sel',m==='lr');
    document.getElementById('mb-ann').classList.toggle('sel',m==='ann');
  }
}

async function predInd(e){
  e.preventDefault();
  const btn=document.getElementById('btn-ind');
  btn.disabled=true; btn.textContent='Procesando...';
  const features=[];
  for(let i=0;i<NCOLS;i++) features.push(parseFloat(document.querySelector(`[name="f${i}"]`).value));
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:modI,features})});
    const res=await r.json();
    if(res.error){alert(res.error);return;}
    showIndResult(res);
  }catch(err){alert('Error: '+err.message);}
  finally{btn.disabled=false;btn.textContent='Predecir';}
}

function showIndResult(res){
  const box=document.getElementById('rbox-ind');
  const cls=res.label.toLowerCase();
  box.className='rbox show '+cls;
  const nombre=modI==='lr'?'Regresion Logistica':'Red Neuronal ANN';
  document.getElementById('r-modelo').textContent='Modelo: '+nombre;
  const labels={'Dropout':'Abandono Escolar','Enrolled':'Matriculado (en curso)','Graduate':'Graduado (Exito)'};
  document.getElementById('r-label').textContent=labels[res.label]||res.label;
  let barsHtml='';
  for(const c of CLASSES){
    const pct=Math.round((res.probabilities[c]||0)*100);
    barsHtml+=`<div class="prow"><span class="plbl">${c}</span><div class="pbar"><div class="pfill" style="width:${pct}%;background:${CLASS_COLORS[c]}"></div></div><span class="pval">${pct}%</span></div>`;
  }
  document.getElementById('r-bars').innerHTML=barsHtml;
}

/* ===== FILE UPLOAD BATCH ===== */
let csvRows = null;
let csvTrueLabels = null;
const TARGET_MAP = {'Dropout':0,'Enrolled':1,'Graduate':2,'dropout':0,'enrolled':1,'graduate':2};

function handleFile(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    parseCSV(e.target.result, file.name);
  };
  reader.readAsText(file);
}

function parseCSV(text, name) {
  const lines = text.trim().split('\n').filter(l => l.trim());
  if (lines.length < 2) { alert('El archivo esta vacio o solo tiene encabezado.'); return; }

  // Detectar si la primera linea es encabezado (contiene texto no numerico)
  const firstLine = lines[0].split(/[,;\t]/);
  let hasHeader = false;
  if (firstLine.some(v => isNaN(v.trim()) && v.trim() !== '')) hasHeader = true;

  const dataLines = hasHeader ? lines.slice(1) : lines;
  const sep = lines[0].includes(';') ? ';' : (lines[0].includes('\t') ? '\t' : ',');
  const parsed = dataLines.map(l => l.split(sep).map(v => v.trim()));

  // Detectar si hay columna Target
  const numCols = parsed[0].length;
  let hasTarget = false;
  csvTrueLabels = null;

  if (hasHeader) {
    const headers = firstLine.map(h => h.trim().toLowerCase());
    const targetIdx = headers.indexOf('target');
    if (targetIdx >= 0) {
      hasTarget = true;
      csvRows = parsed.map(r => {
        const row = [...r];
        row.splice(targetIdx, 1);
        return row.map(Number);
      });
      csvTrueLabels = parsed.map(r => {
        const val = r[targetIdx];
        return TARGET_MAP[val] !== undefined ? TARGET_MAP[val] : parseInt(val);
      });
    }
  }

  if (!hasTarget && numCols === 37) {
    csvRows = parsed.map(r => r.slice(0, 36).map(Number));
    csvTrueLabels = parsed.map(r => {
      const val = r[36];
      return TARGET_MAP[val] !== undefined ? TARGET_MAP[val] : parseInt(val);
    });
  } else if (!hasTarget) {
    csvRows = parsed.map(r => r.map(Number));
  }

  // Mostrar info del archivo
  document.getElementById('file-info').style.display = 'flex';
  document.getElementById('file-name').textContent = name;
  document.getElementById('file-rows').textContent = csvRows.length + ' estudiantes' + (csvTrueLabels ? ' (con Target)' : '');
  document.getElementById('btn-batch').disabled = false;
  document.getElementById('upload-zone').style.display = 'none';
}

async function predBatch() {
  if (!csvRows || csvRows.length === 0) { alert('Sube un archivo CSV primero.'); return; }
  const btn = document.getElementById('btn-batch');
  btn.disabled = true; btn.textContent = 'Procesando...';
  try {
    const body = { model: modB, rows: csvRows };
    if (csvTrueLabels) body.true_labels = csvTrueLabels;
    const r = await fetch('/predict_batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const res = await r.json();
    if (res.error) { alert(res.error); return; }
    showBatchResults(res, csvRows, csvTrueLabels);
  } catch (err) { alert('Error: ' + err.message); }
  finally { btn.disabled = false; btn.textContent = '\u{1F4CA} Predecir Lote'; }
}

function showBatchResults(res, rows, trueLabels) {
  const total = res.predictions.length;
  const counts = { Dropout: 0, Enrolled: 0, Graduate: 0 };
  res.labels.forEach(l => counts[l]++);
  let mhtml = '<div class="met"><div class="mv">' + total + '</div><div class="ml">Total</div></div>';
  mhtml += '<div class="met"><div class="mv" style="color:var(--red)">' + counts.Dropout + '</div><div class="ml">Abandono</div></div>';
  mhtml += '<div class="met"><div class="mv" style="color:var(--amber)">' + counts.Enrolled + '</div><div class="ml">Matriculados</div></div>';
  mhtml += '<div class="met"><div class="mv" style="color:var(--green)">' + counts.Graduate + '</div><div class="ml">Graduados</div></div>';
  if (res.accuracy !== undefined) mhtml += '<div class="met"><div class="mv">' + Math.round(res.accuracy * 100) + '%</div><div class="ml">Accuracy</div></div>';
  document.getElementById('batch-mets').innerHTML = mhtml;

  const cmSec = document.getElementById('cm-section');
  if (res.confusion_matrix) {
    cmSec.style.display = 'block';
    let cmHtml = '<div class="cm-cell"></div>';
    CLASSES.forEach(c => cmHtml += '<div class="cm-cell cm-head">' + c + '</div>');
    for (let i = 0; i < 3; i++) {
      cmHtml += '<div class="cm-cell cm-head">' + CLASSES[i] + '</div>';
      for (let j = 0; j < 3; j++) {
        const v = res.confusion_matrix[i][j];
        const maxV = Math.max(...res.confusion_matrix.flat());
        const opacity = Math.min(v / maxV * 0.6 + 0.1, 0.7);
        const color = i === j ? '22,163,74' : '220,38,38';
        cmHtml += `<div class="cm-cell cm-val" style="background:rgba(${color},${opacity})">${v}</div>`;
      }
    }
    document.getElementById('cm-grid').innerHTML = cmHtml;

    let metHtml = '';
    if (res.metrics) {
      for (const c of CLASSES) {
        if (!res.metrics[c]) continue;
        const m = res.metrics[c];
        metHtml += `<tr><td><span class="badge ${BADGE_CLS[c]}">${c}</span></td><td>${(m.precision * 100).toFixed(1)}%</td><td>${(m.recall * 100).toFixed(1)}%</td><td>${(m['f1-score'] * 100).toFixed(1)}%</td><td>${m.support}</td></tr>`;
      }
    }
    document.getElementById('met-body').innerHTML = metHtml;
  } else { cmSec.style.display = 'none'; }

  document.getElementById('tbl-head').innerHTML = '<th>#</th><th>Prediccion</th><th>P(Drop)</th><th>P(Enr)</th><th>P(Grad)</th>';
  let thtml = '';
  for (let i = 0; i < Math.min(rows.length, 100); i++) {
    const lbl = res.labels[i];
    const badge = `<span class="badge ${BADGE_CLS[lbl]}">${lbl}</span>`;
    const ps = res.probabilities[i];
    thtml += `<tr><td>${i + 1}</td><td>${badge}</td><td>${(ps[0] * 100).toFixed(1)}%</td><td>${(ps[1] * 100).toFixed(1)}%</td><td>${(ps[2] * 100).toFixed(1)}%</td></tr>`;
  }
  if (rows.length > 100) thtml += '<tr><td colspan="5" style="text-align:center;color:var(--muted)">...mostrando primeras 100 filas</td></tr>';
  document.getElementById('tbl-body').innerHTML = thtml;
  const out = document.getElementById('batch-out');
  out.style.display = 'block'; out.scrollIntoView({ behavior: 'smooth' });
}

function clearBatch() {
  csvRows = null;
  csvTrueLabels = null;
  document.getElementById('csv-file').value = '';
  document.getElementById('file-info').style.display = 'none';
  document.getElementById('upload-zone').style.display = '';
  document.getElementById('btn-batch').disabled = true;
  document.getElementById('batch-out').style.display = 'none';
}

/* Drag and drop */
document.addEventListener('DOMContentLoaded', function() {
  const zone = document.getElementById('upload-zone');
  if (!zone) return;
  zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', function() { zone.classList.remove('dragover'); });
  zone.addEventListener('drop', function(e) {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      document.getElementById('csv-file').files = e.dataTransfer.files;
      const reader = new FileReader();
      reader.onload = function(ev) { parseCSV(ev.target.result, file.name); };
      reader.readAsText(file);
    } else { alert('Por favor sube un archivo .csv'); }
  });
});

/* ===== METRICS / CONFUSION MATRIX ===== */

async function loadMetrics(panel) {
  const modelo = panel === 'ind' ? modI : modB;
  const btn = document.getElementById('btn-metrics-' + panel);
  const outDiv = document.getElementById('metrics-' + panel + '-out');
  btn.disabled = true;
  btn.textContent = 'Cargando...';
  try {
    const r = await fetch('/metrics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelo })
    });
    const res = await r.json();
    if (res.error) { alert(res.error); return; }
    renderMetrics(res, panel);
    outDiv.style.display = 'block';
    outDiv.scrollIntoView({ behavior: 'smooth' });
  } catch (err) { alert('Error: ' + err.message); }
  finally {
    btn.disabled = false;
    btn.innerHTML = '&#128202; Actualizar Matriz';
  }
}

function renderMetrics(res, panel) {
  // Summary cards
  const acc = Math.round(res.accuracy * 100);
  let shtml = `<div class="met"><div class="mv">${acc}%</div><div class="ml">Accuracy</div></div>`;
  for (const c of CLASSES) {
    if (!res.metrics[c]) continue;
    const m = res.metrics[c];
    shtml += `<div class="met"><div class="mv" style="color:${CLASS_COLORS[c]}">${(m['f1-score']*100).toFixed(1)}%</div><div class="ml">F1 ${c}</div></div>`;
  }
  document.getElementById('metrics-' + panel + '-summary').innerHTML = shtml;

  // Confusion matrix
  const cm = res.confusion_matrix;
  const maxV = Math.max(...cm.flat());
  let cmHtml = '<div class="cm-cell cm-corner"><span class="cm-axis-label">Real \\ Pred</span></div>';
  CLASSES.forEach(c => cmHtml += `<div class="cm-cell cm-head">${c}</div>`);
  for (let i = 0; i < 3; i++) {
    cmHtml += `<div class="cm-cell cm-head">${CLASSES[i]}</div>`;
    for (let j = 0; j < 3; j++) {
      const v = cm[i][j];
      const rowSum = cm[i].reduce((a,b)=>a+b,0);
      const opacity = maxV > 0 ? Math.min(v / maxV * 0.65 + 0.08, 0.75) : 0.08;
      const color = i === j ? '22,163,74' : '220,38,38';
      const pct = rowSum > 0 ? Math.round(v / rowSum * 100) : 0;
      cmHtml += `<div class="cm-cell cm-val" style="background:rgba(${color},${opacity})"><span class="cm-num">${v}</span><span class="cm-pct">${pct}%</span></div>`;
    }
  }
  document.getElementById('cm-' + panel).innerHTML = cmHtml;

  // Per-class metrics table
  let metHtml = '';
  for (const c of CLASSES) {
    if (!res.metrics[c]) continue;
    const m = res.metrics[c];
    metHtml += `<tr><td><span class="badge ${BADGE_CLS[c]}">${c}</span></td><td>${(m.precision*100).toFixed(1)}%</td><td>${(m.recall*100).toFixed(1)}%</td><td>${(m['f1-score']*100).toFixed(1)}%</td><td>${m.support}</td></tr>`;
  }
  document.getElementById('met-' + panel + '-body').innerHTML = metHtml;
}
