/**
 * PROCTIFY — Exam UI: timer, status polling, submit, termination overlay.
 */
(function () {
  const timerEl = document.getElementById("timer");
  const stFace = document.getElementById("st-face");
  const stEye = document.getElementById("st-eye");
  const stHead = document.getElementById("st-head");
  const stObj = document.getElementById("st-obj");
  const stTrust = document.getElementById("st-trust");
  const warnBanner = document.getElementById("warn-banner");
  const alertLine = document.getElementById("alert-line");
  const fpsVal = document.getElementById("fps-val");
  const warnLog = document.getElementById("warn-log");
  const freeze = document.getElementById("freeze-overlay");
  const freezeReason = document.getElementById("freeze-reason");
  const btnSubmit = document.getElementById("btn-submit");

  const started = Date.now();

  function formatMs(ms) {
    const s = Math.floor(ms / 1000);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return [h, m, sec].map((n) => String(n).padStart(2, "0")).join(":");
  }

  setInterval(() => {
    timerEl.textContent = formatMs(Date.now() - started);
  }, 250);

  function stripBracket(line) {
    if (!line) return "—";
    return line.replace(/^\[[^:]+:\s*/, "").replace(/\]$/, "");
  }

  function poll() {
    fetch("/api/exam_state")
      .then((r) => r.json())
      .then((d) => {
        stFace.textContent = stripBracket(d.face_status);
        stEye.textContent = stripBracket(d.eye_status);
        stHead.textContent = stripBracket(d.head_status);
        stObj.textContent = stripBracket(d.object_status);
        stTrust.textContent = d.trust_score + "%";

        const bad =
          /Not Detected|Multiple|Phone|Away|Down/i.test(d.face_status + d.object_status);
        stFace.className = "val " + (bad ? "bad" : "ok");
        stObj.className = "val " + (d.object_status && d.object_status.includes("Phone") ? "bad" : "warn");

        warnBanner.textContent = d.warnings || "";
        alertLine.textContent = d.alert || "";
        fpsVal.textContent = d.fps;

        warnLog.innerHTML = "";
        (d.warning_log || []).slice(-12).forEach((w) => {
          const li = document.createElement("li");
          li.textContent = (w.t ? w.t + " — " : "") + w.text;
          warnLog.appendChild(li);
        });

        if (d.terminated) {
          freeze.classList.remove("hidden");
          freezeReason.textContent =
            d.terminate_reason || "Cheating or policy violation detected.";
        }
      })
      .catch(() => {});
  }

  setInterval(poll, 400);
  poll();

  btnSubmit.addEventListener("click", () => {
    fetch("/end_exam", { method: "POST" })
      .then(() => {
        window.location.href = "/report";
      })
      .catch(() => {
        window.location.href = "/report";
      });
  });
})();
