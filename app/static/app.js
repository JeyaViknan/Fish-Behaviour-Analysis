(function () {
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const resultsSection = document.getElementById('resultsSection');
  const loading = document.getElementById('loading');
  const errorSection = document.getElementById('error');
  const errorMessage = document.getElementById('errorMessage');
  const previewImg = document.getElementById('previewImg');
  const overlay = document.getElementById('overlay');
  const summaryEl = document.getElementById('summary');
  const meaningsList = document.getElementById('meaningsList');
  const legend = document.getElementById('legend');

  const API_BASE = window.location.origin;

  function showLoading(show) {
    loading.hidden = !show;
    if (show) {
      errorSection.hidden = true;
      resultsSection.hidden = true;
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorSection.hidden = false;
    resultsSection.hidden = true;
    loading.hidden = true;
  }

  function drawDetections(img, detections) {
    const rect = previewImg.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;

    overlay.style.width = rect.width + 'px';
    overlay.style.height = rect.height + 'px';
    overlay.width = rect.width;
    overlay.height = rect.height;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    detections.forEach(function (d) {
      const [x1, y1, x2, y2] = d.bbox;
      const sx1 = (x1 / scaleX);
      const sy1 = (y1 / scaleY);
      const sx2 = (x2 / scaleX);
      const sy2 = (y2 / scaleY);
      const w = sx2 - sx1;
      const h = sy2 - sy1;

      ctx.strokeStyle = d.color || '#38bdf8';
      ctx.lineWidth = 2;
      ctx.strokeRect(sx1, sy1, w, h);

      ctx.fillStyle = d.color || '#38bdf8';
      ctx.font = '12px system-ui, sans-serif';
      const label = d.label + ' ' + (Math.round(d.confidence * 100) + '%');
      const tw = ctx.measureText(label).width;
      ctx.fillRect(sx1, sy1 - 18, tw + 8, 18);
      ctx.fillStyle = '#0f172a';
      ctx.fillText(label, sx1 + 4, sy1 - 5);
    });
  }

  function renderMeanings(detections, behaviorMeanings) {
    const byClass = {};
    detections.forEach(function (d) {
      byClass[d.class_id] = byClass[d.class_id] || {
        label: d.label,
        meaning: d.meaning,
        is_anomaly: d.is_anomaly,
        color: d.color,
        count: 0
      };
      byClass[d.class_id].count += 1;
    });

    const order = [0, 1, 2, 3, 4];
    const classNames = ['normal', 'ph', 'low-temp', 'high-temp', 'hypoxia'];

    meaningsList.innerHTML = '';

    order.forEach(function (classId, i) {
      const info = byClass[classId] || behaviorMeanings[classId];
      if (!info) return;

      const card = document.createElement('div');
      card.className = 'meaning-card ' + (info.is_anomaly ? 'anomaly ' + classNames[classId] : 'normal');
      const count = (byClass[classId] && byClass[classId].count) ? ' (' + byClass[classId].count + ')' : '';
      card.innerHTML =
        '<div class="label">' + escapeHtml(info.label) + count + '</div>' +
        '<p class="meaning">' + escapeHtml(info.meaning) + '</p>' +
        '<span class="badge ' + (info.is_anomaly ? 'anomaly' : 'normal') + '">' +
        (info.is_anomaly ? 'Anomaly' : 'Normal') + '</span>';
      meaningsList.appendChild(card);
    });

    legend.innerHTML =
      '<h3>Legend</h3><div class="legend-items">' +
      Object.entries(behaviorMeanings).map(function (e) {
        const id = e[0];
        const m = e[1];
        return '<div class="legend-item"><span style="background:' + m.color + '"></span> ' + escapeHtml(m.label) + '</div>';
      }).join('') +
      '</div>';
  }

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  function runDetection(file) {
    showLoading(true);
    const form = new FormData();
    form.append('image', file);

    fetch(API_BASE + '/api/detect', {
      method: 'POST',
      body: form
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        showLoading(false);
        if (data.error) {
          showError(data.error);
          return;
        }

        const detections = data.detections || [];
        const behaviorMeanings = data.behavior_meanings || {};
        const anomaliesDetected = data.anomalies_detected;

        const reader = new FileReader();
        reader.onload = function () {
          previewImg.src = reader.result;
          previewImg.onload = function () {
            drawDetections(previewImg, detections);
          };
        };
        reader.readAsDataURL(file);

        if (anomaliesDetected) {
          summaryEl.textContent = 'Anomalous behavior(s) detected. See behavior meanings below.';
          summaryEl.classList.add('anomaly');
        } else if (detections.length) {
          summaryEl.textContent = detections.length + ' fish detected (normal behavior).';
          summaryEl.classList.remove('anomaly');
        } else {
          summaryEl.textContent = 'No fish detected.';
          summaryEl.classList.remove('anomaly');
        }

        renderMeanings(detections, behaviorMeanings);
        resultsSection.hidden = false;
      })
      .catch(function (err) {
        showLoading(false);
        showError('Detection failed: ' + (err.message || 'Network error'));
      });
  }

  browseBtn.addEventListener('click', function () { fileInput.click(); });
  fileInput.addEventListener('change', function () {
    const file = fileInput.files[0];
    if (file && file.type.startsWith('image/')) runDetection(file);
  });

  uploadZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', function () {
    uploadZone.classList.remove('dragover');
  });
  uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) runDetection(file);
  });
})();
