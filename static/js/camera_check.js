/**
 * PROCTIFY — Camera check: capture frames and verify single face via API.
 */
(function () {
  const video = document.getElementById("preview");
  const canvas = document.getElementById("snap");
  const statusEl = document.getElementById("cam-status");
  const verifyEl = document.getElementById("verify-msg");
  const startBtn = document.getElementById("start-exam");

  let okStreak = 0;
  const needStreak = 6;
  let stopped = false;

  /** Wait until video has dimensions (required on Chrome / Edge before capture). */
  function waitForVideoReady() {
    return new Promise(function (resolve, reject) {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        resolve();
        return;
      }
      const onMeta = function () {
        cleanup();
        resolve();
      };
      const onErr = function () {
        cleanup();
        reject(new Error("Video failed to load"));
      };
      function cleanup() {
        video.removeEventListener("loadedmetadata", onMeta);
        video.removeEventListener("error", onErr);
        clearTimeout(tid);
      }
      video.addEventListener("loadedmetadata", onMeta, { once: true });
      video.addEventListener("error", onErr, { once: true });
      const tid = setTimeout(function () {
        cleanup();
        reject(new Error("Camera preview timed out — allow camera access and refresh."));
      }, 15000);
    });
  }

  async function setupCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      video.setAttribute("playsinline", "true");
      video.muted = true;
      video.srcObject = stream;
      statusEl.textContent = "Starting camera preview…";

      await video.play().catch(function () {});
      await waitForVideoReady();

      statusEl.textContent = "Camera active — verifying…";
      tick();
    } catch (e) {
      statusEl.textContent =
        "Could not access camera: " + (e && e.message ? e.message : String(e));
      verifyEl.textContent =
        "Allow camera permission for this site, use HTTPS or localhost, and try again.";
      verifyEl.className = "verify-result bad";
    }
  }

  function tick() {
    if (stopped) return;

    var w = video.videoWidth;
    var h = video.videoHeight;
    if (!w || !h) {
      statusEl.textContent = "Waiting for video frame…";
      setTimeout(tick, 200);
      return;
    }

    // Downscale before upload (faster API + MediaPipe)
    var maxW = 640;
    var scale = w > maxW ? maxW / w : 1;
    var cw = Math.round(w * scale);
    var ch = Math.round(h * scale);
    canvas.width = cw;
    canvas.height = ch;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, cw, ch);
    var dataUrl = canvas.toDataURL("image/jpeg", 0.8);

    fetch("/api/camera_verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    })
      .then(function (r) {
        if (!r.ok) {
          return r.text().then(function (t) {
            throw new Error("Server " + r.status + ": " + (t || r.statusText));
          });
        }
        return r.json();
      })
      .then(function (data) {
        if (data.ok) {
          okStreak += 1;
          verifyEl.textContent = data.message || "Camera Verified ✅";
          verifyEl.className = "verify-result ok";
          if (okStreak >= needStreak) {
            startBtn.classList.remove("disabled");
            statusEl.textContent = "Ready";
            stopped = true;
            return;
          }
        } else {
          okStreak = 0;
          verifyEl.textContent = data.message || "Adjust your position";
          verifyEl.className = "verify-result bad";
        }
        setTimeout(tick, 350);
      })
      .catch(function (err) {
        verifyEl.textContent =
          "Verify request failed: " + (err && err.message ? err.message : String(err));
        verifyEl.className = "verify-result bad";
        setTimeout(tick, 1200);
      });
  }

  setupCamera();
})();
