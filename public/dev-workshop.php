<?php
/**
 * Pyx Dev Workshop — review applications & reply (staff; same password as trainer gate).
 */
declare(strict_types=1);

require_once __DIR__ . '/workforpyx_lib.php';

$tracks = workforpyx_tracks();
$flash = (string) ($_GET['flash'] ?? '');
$view_id = trim((string) ($_GET['id'] ?? ''));
$action = (string) ($_GET['action'] ?? '');
$tab = (string) ($_GET['tab'] ?? 'apps');
if ($tab !== 'traffic') {
    $tab = 'apps';
}

// Resume download (staff session checked in browser before page load; file served here)
if ($action === 'resume' && $view_id !== '') {
    $app = workforpyx_find_by_id($view_id);
    if ($app && !empty($app['resume_stored'])) {
        $path = workforpyx_data_dir() . '/resumes/' . basename((string) $app['resume_stored']);
        if (is_file($path)) {
            $mime = 'application/octet-stream';
            $ext = strtolower(pathinfo($path, PATHINFO_EXTENSION));
            if ($ext === 'pdf') {
                $mime = 'application/pdf';
            } elseif (in_array($ext, ['doc', 'docx'], true)) {
                $mime = 'application/msword';
            } elseif ($ext === 'txt') {
                $mime = 'text/plain';
            }
            header('Content-Type: ' . $mime);
            header('Content-Disposition: attachment; filename="' . basename($path) . '"');
            header('Content-Length: ' . (string) filesize($path));
            readfile($path);
            exit;
        }
    }
    http_response_code(404);
    echo 'Resume not found.';
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $post_action = (string) ($_POST['action'] ?? '');
    $post_id = trim((string) ($_POST['id'] ?? ''));
    if ($post_action === 'reply' && $post_id !== '') {
        $reply = trim((string) ($_POST['reply_body'] ?? ''));
        if ($reply === '') {
            $flash = 'err:Write a message before sending.';
        } elseif (workforpyx_add_reply($post_id, $reply, 'dev')) {
            $flash = 'ok:Reply saved. (Email the applicant separately using their address on file.)';
            $view_id = $post_id;
        } else {
            $flash = 'err:Could not save reply.';
        }
    }
}

$applications = workforpyx_load_applications();
usort($applications, static function ($a, $b) {
    return strcmp((string) ($b['created'] ?? ''), (string) ($a['created'] ?? ''));
});

$detail = $view_id !== '' ? workforpyx_find_by_id($view_id) : null;

function workforpyx_status_label(string $status): string
{
    return match ($status) {
        'hired' => 'Hired',
        'rejected' => 'Rejected',
        'replied' => 'Replied',
        'reviewing' => 'Reviewing',
        'new' => 'New',
        default => ucfirst($status),
    };
}

function workforpyx_status_pill_class(string $status): string
{
    return match ($status) {
        'hired' => 'pill--hired',
        'rejected' => 'pill--rejected',
        'replied' => 'pill--replied',
        default => 'pill--new',
    };
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pyx Dev Workshop</title>
  <link rel="icon" href="/brand/pyx-app-icon.png" type="image/png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --ok: #34d399;
      --warn: #fbbf24;
      --err: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Plus Jakarta Sans", system-ui, sans-serif;
      background: #0f172a;
      color: var(--text);
      line-height: 1.5;
    }
    .wrap { max-width: 960px; margin: 0 auto; padding: 20px 16px 40px; }
    h1 { margin: 0 0 4px; font-size: 1.5rem; }
    .sub { color: var(--muted); margin: 0 0 20px; font-size: 0.9rem; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 16px; }
    .btn {
      display: inline-block;
      padding: 8px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #334155;
      color: var(--text);
      font: inherit;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
    }
    .btn-primary { background: #2563eb; border-color: #2563eb; }
    .btn-ghost { background: transparent; }
    .flash { padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 0.9rem; }
    .flash--ok { background: rgba(52, 211, 153, 0.15); color: #a7f3d0; }
    .flash--err { background: rgba(248, 113, 113, 0.15); color: #fecaca; }
    .grid { display: grid; grid-template-columns: 1fr 1.2fr; gap: 16px; align-items: start; }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }
    .panel h2 { margin: 0 0 10px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; }
    tr:hover td { background: rgba(56, 189, 248, 0.06); }
    tr.is-active td { background: rgba(99, 102, 241, 0.15); }
    a.row-link { color: inherit; text-decoration: none; display: block; }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 700;
    }
    .pill--new { background: rgba(56, 189, 248, 0.2); color: #7dd3fc; }
    .pill--replied { background: rgba(52, 211, 153, 0.2); color: #6ee7b7; }
    .pill--hired { background: rgba(52, 211, 153, 0.28); color: #6ee7b7; }
    .pill--rejected { background: rgba(248, 113, 113, 0.2); color: #fca5a5; }
    .decision-box {
      margin-top: 18px; padding: 14px; border-radius: 12px;
      border: 1px solid rgba(129, 140, 248, 0.35); background: rgba(15, 23, 42, 0.6);
    }
    .decision-box h2 { margin-top: 0; }
    .decision-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .btn-hired { background: #166534; border-color: #166534; color: #fff; }
    .btn-rejected { background: #9f1239; border-color: #9f1239; color: #fff; }
    .field { margin-bottom: 12px; }
    .field dt { font-size: 0.72rem; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }
    .field dd { margin: 0; white-space: pre-wrap; word-break: break-word; }
    .replies { margin-top: 16px; }
    .reply {
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0f172a;
      margin-bottom: 8px;
      font-size: 0.88rem;
    }
    .reply meta { display: block; font-size: 0.72rem; color: var(--muted); margin-bottom: 4px; }
    textarea {
      width: 100%;
      min-height: 100px;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font: inherit;
      margin-bottom: 8px;
    }
    .features {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }
    .feature {
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      font-size: 0.82rem;
    }
    .feature.is-on { border-color: var(--accent); color: #7dd3fc; }
    .feature.is-soon { opacity: 0.5; }
    #gate { text-align: center; padding: 48px 20px; }
    .workshop-tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .workshop-tabs a {
      padding: 8px 14px;
      border-radius: 999px;
      border: 1px solid var(--border);
      text-decoration: none;
      color: var(--muted);
      font-size: 0.88rem;
      font-weight: 700;
    }
    .workshop-tabs a.is-active {
      color: #e0f2fe;
      border-color: rgba(56, 189, 248, 0.55);
      background: rgba(14, 165, 233, 0.15);
    }
    .traffic-layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    @media (max-width: 800px) {
      .traffic-layout { grid-template-columns: 1fr; }
    }
    .traffic-swatch {
      width: 100%;
      height: 120px;
      border-radius: 12px;
      border: 2px solid var(--border);
      background: #334155;
      margin: 12px 0;
    }
    .traffic-result-label { font-size: 1.1rem; font-weight: 700; margin: 0; }
    .traffic-result-meta { font-size: 0.82rem; color: var(--muted); margin: 4px 0 0; }
    .traffic-url-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
    .traffic-url-row input {
      flex: 1;
      min-width: 200px;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font: inherit;
    }
    #trafficPreview {
      max-width: 100%;
      max-height: 200px;
      border-radius: 8px;
      border: 1px solid var(--border);
      margin-top: 8px;
    }
    .traffic-starters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .traffic-starter {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 6px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0f172a;
      cursor: pointer;
      max-width: 100px;
      font: inherit;
      color: var(--muted);
      font-size: 0.7rem;
    }
    .traffic-starter img {
      width: 88px;
      height: 60px;
      object-fit: cover;
      border-radius: 6px;
    }
    .traffic-train-btns { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
    .btn-xs { padding: 5px 10px; font-size: 0.78rem; }
    .traffic-train-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .traffic-train-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: #0f172a;
    }
    .traffic-train-card img {
      width: 100%;
      height: 80px;
      object-fit: cover;
      display: block;
    }
    .traffic-train-card__body {
      padding: 6px 8px;
      font-size: 0.78rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
    }
    .traffic-signal-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }
    .traffic-muted { color: var(--muted); font-size: 0.88rem; }
    #trafficLog {
      max-height: 160px;
      overflow: auto;
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 12px;
      border-top: 1px solid var(--border);
      padding-top: 8px;
    }
    .traffic-input-tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
    }
    .traffic-input-tabs button {
      font: inherit;
      font-weight: 700;
      padding: 8px 14px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--muted);
      cursor: pointer;
    }
    .traffic-input-tabs button.is-active {
      color: #e0f2fe;
      border-color: rgba(56, 189, 248, 0.5);
      background: rgba(14, 165, 233, 0.12);
    }
    .traffic-live-wrap {
      position: relative;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #020617;
      margin: 10px 0;
    }
    #trafficLiveVideo {
      width: 100%;
      max-height: 280px;
      display: block;
      object-fit: contain;
    }
    .traffic-live-swatch {
      height: 48px;
      border-top: 1px solid var(--border);
    }
    .traffic-live-label {
      padding: 8px 12px;
      font-size: 0.88rem;
      font-weight: 700;
      margin: 0;
    }
    .traffic-live-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 10px 0;
    }
    .traffic-live-controls label {
      font-size: 0.78rem;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .traffic-live-controls input[type="number"] {
      width: 4rem;
      padding: 4px 6px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font: inherit;
    }
    .traffic-roadmap {
      padding: 12px;
      border-radius: 10px;
      border: 1px dashed rgba(56, 189, 248, 0.35);
      background: rgba(14, 165, 233, 0.06);
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 14px;
    }
    .traffic-train-card__live {
      height: 80px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      color: #64748b;
      background: #1e293b;
    }
    .traffic-web-search {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
      align-items: center;
    }
    .traffic-web-search input[type="search"] {
      flex: 1;
      min-width: 200px;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font: inherit;
    }
    .traffic-web-presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .traffic-web-presets button {
      font: inherit;
      font-size: 0.78rem;
      font-weight: 700;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: #cbd5e1;
      cursor: pointer;
    }
    .traffic-web-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .traffic-web-card {
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      background: #0f172a;
    }
    .traffic-web-card img {
      width: 100%;
      height: 140px;
      object-fit: cover;
      display: block;
      background: #1e293b;
    }
    .traffic-web-card__body { padding: 8px 10px 10px; }
    .traffic-web-card__title {
      font-size: 0.72rem;
      color: var(--muted);
      margin: 0 0 8px;
      line-height: 1.35;
      max-height: 2.7em;
      overflow: hidden;
    }
    .traffic-web-card__btns { display: flex; flex-wrap: wrap; gap: 4px; }
    .traffic-stats {
      font-size: 0.82rem;
      color: #a5b4fc;
      margin: 0 0 12px;
    }
  </style>
</head>
<body>
  <div id="gate">
    <p>Checking Dev Workshop access…</p>
    <p class="sub"><a href="/pyx-trainer-auth.html?next=/dev-workshop.php">Sign in</a></p>
  </div>

  <div id="workshop" hidden>
    <div class="wrap">
      <div class="toolbar">
        <h1 style="margin:0;flex:1;">Pyx Dev Workshop</h1>
        <button type="button" class="btn btn-ghost" id="lockWorkshop">Lock</button>
        <a class="btn btn-ghost" href="/pyx-firebase-trainer.html">Trainer</a>
        <a class="btn btn-ghost" href="/workforpyx.php">Public apply form</a>
        <a class="btn btn-ghost" href="/">Studio</a>
      </div>
      <p class="sub">Staff tools for Pyx — same password as trainer sign-in. More features can plug in here later.</p>

      <nav class="workshop-tabs" aria-label="Workshop sections">
        <a href="/dev-workshop.php?tab=apps" class="<?php echo $tab === 'apps' ? 'is-active' : ''; ?>">Applications</a>
        <a href="/dev-workshop.php?tab=traffic" class="<?php echo $tab === 'traffic' ? 'is-active' : ''; ?>">Traffic lights</a>
      </nav>

      <div class="features" aria-label="Workshop features">
        <span class="feature <?php echo $tab === 'apps' ? 'is-on' : 'is-soon'; ?>">Applications inbox</span>
        <span class="feature <?php echo $tab === 'apps' ? 'is-on' : 'is-soon'; ?>">Hired / rejected emails</span>
        <span class="feature <?php echo $tab === 'traffic' ? 'is-on' : 'is-soon'; ?>">Traffic light analyzer</span>
      </div>

      <?php if ($tab === 'traffic'): ?>
      <section class="panel" id="tab-traffic">
        <h2>Traffic light analyzer</h2>
        <p class="traffic-muted" style="margin:0 0 14px;">
          <strong>Train from web</strong> — Pyx searches for traffic-light photos, hosts them on this site, and you label each one.
          The model learns from your labels (not guesses alone). Then analyze or use live preview.
        </p>
        <p class="traffic-stats" id="trafficStats">Loading training stats…</p>
        <div class="traffic-input-tabs">
          <button type="button" class="is-active" id="trafficTabTrainWeb">Train from web</button>
          <button type="button" id="trafficTabImage">Test image</button>
          <button type="button" id="trafficTabLive">Live video</button>
        </div>
        <div class="traffic-layout">
          <div>
            <div id="trafficPanelTrainWeb">
              <h3 style="margin:0 0 8px;font-size:1rem;">Search &amp; label images</h3>
              <p class="traffic-muted" style="margin:0 0 10px;font-size:0.85rem;">
                Images are downloaded to Pyx and shown below (public URLs on this site). Click the color you see on the signal.
              </p>
              <div class="traffic-web-search">
                <input type="search" id="trafficWebQuery" placeholder="e.g. green traffic light close up" value="green traffic light" />
                <button type="button" class="btn btn-primary" id="trafficWebSearchBtn">Search web</button>
              </div>
              <div class="traffic-web-presets" id="trafficWebPresets"></div>
              <div class="traffic-web-grid" id="trafficWebGrid">
                <p class="traffic-muted">Search to load training images.</p>
              </div>
            </div>

            <div id="trafficPanelImage" hidden>
            <label for="trafficImageUrl" style="font-size:0.85rem;font-weight:600;">Image URL (https)</label>
            <div class="traffic-url-row">
              <input type="url" id="trafficImageUrl" placeholder="https://…/traffic-light.jpg" />
              <button type="button" class="btn btn-primary" id="trafficAnalyzeBtn">Analyze</button>
              <button type="button" class="btn" id="trafficSendBtn">Send color</button>
            </div>
            <img id="trafficPreview" alt="Preview" hidden />
            <div class="traffic-starters" id="trafficStarters"></div>
            <div class="traffic-swatch" id="trafficSwatch" aria-hidden="true"></div>
            <p class="traffic-result-label" id="trafficResultLabel">No analysis yet</p>
            <p class="traffic-result-meta" id="trafficResultMeta"></p>
            <p class="traffic-muted" style="font-size:0.78rem;margin-top:10px;">
              Integrate: listen for <code>pyx-traffic-color</code> on <code>window</code>, or
              <code>postMessage({ type: 'pyx-traffic-color', hex, color, mode, frame_id })</code>.
            </p>
            </div>

            <div id="trafficPanelLive" hidden>
              <h3 style="margin:0 0 8px;font-size:1rem;">Live video (preview)</h3>
              <p class="traffic-muted" style="margin:0 0 10px;font-size:0.85rem;">
                Point your camera at a signal or load a clip. Frames run through the same analyzer as still images (~5 fps default).
              </p>
              <div class="traffic-live-controls">
                <button type="button" class="btn btn-primary" id="trafficLiveCamera">Start camera</button>
                <input type="file" id="trafficLiveFile" accept="video/*,image/*" hidden />
                <button type="button" class="btn" id="trafficLiveFileBtn">Use video file</button>
                <button type="button" class="btn btn-ghost" id="trafficLiveStop" disabled>Stop</button>
                <label>FPS cap <input type="number" id="trafficLiveFps" min="1" max="15" value="5" /></label>
                <label>Hold ms <input type="number" id="trafficLiveHold" min="0" max="3000" value="400" title="Min time before re-emitting same color" /></label>
                <label><input type="checkbox" id="trafficLiveAutoEmit" checked /> Auto-send color</label>
              </div>
              <div class="traffic-live-wrap" id="trafficLiveWrap" hidden>
                <video id="trafficLiveVideo" playsinline muted autoplay></video>
                <div class="traffic-live-swatch" id="trafficLiveSwatch"></div>
                <p class="traffic-live-label" id="trafficLiveLabel">Waiting for frames…</p>
              </div>
              <p class="traffic-muted" id="trafficLiveStat" style="font-size:0.78rem;margin:8px 0 0;"></p>
              <p class="traffic-muted" style="font-size:0.82rem;margin:12px 0 6px;">Train from current live frame:</p>
              <div class="traffic-train-btns">
                <button type="button" class="btn btn-xs" id="trafficLiveTrain_red" style="border-color:#ef4444">Red</button>
                <button type="button" class="btn btn-xs" id="trafficLiveTrain_yellow" style="border-color:#eab308">Yellow</button>
                <button type="button" class="btn btn-xs" id="trafficLiveTrain_green" style="border-color:#22c55e">Green</button>
                <button type="button" class="btn btn-xs" id="trafficLiveTrain_off">Off</button>
                <button type="button" class="btn btn-xs" id="trafficLiveTrain_unknown">Unknown</button>
              </div>
            </div>
          </div>
          <div>
            <h3 style="margin:0 0 8px;font-size:1rem;">Saved training set</h3>
            <p class="traffic-muted" style="margin:0 0 8px;">
              Labels you saved (web search, URL, or live frame):
            </p>
            <div class="traffic-train-btns">
              <button type="button" class="btn btn-xs" id="trafficTrain_red" style="border-color:#ef4444">Red</button>
              <button type="button" class="btn btn-xs" id="trafficTrain_yellow" style="border-color:#eab308">Yellow</button>
              <button type="button" class="btn btn-xs" id="trafficTrain_green" style="border-color:#22c55e">Green</button>
              <button type="button" class="btn btn-xs" id="trafficTrain_off">Off</button>
              <button type="button" class="btn btn-xs" id="trafficTrain_unknown">Unknown</button>
            </div>
            <div id="trafficTrainGrid" class="traffic-train-grid"></div>
            <div id="trafficLog"></div>
          </div>
        </div>
      </section>
      <?php else: ?>

      <?php
      if ($flash !== '') {
          $parts = explode(':', $flash, 2);
          $kind = $parts[0] ?? 'ok';
          $msg = $parts[1] ?? $flash;
          $cls = $kind === 'err' ? 'flash--err' : 'flash--ok';
          echo '<div class="flash ' . $cls . '">' . workforpyx_escape($msg) . '</div>';
      }
      ?>

      <div class="grid">
        <div class="panel">
          <h2>Applications (<?php echo count($applications); ?>)</h2>
          <?php if (!$applications): ?>
            <p style="color:var(--muted);font-size:0.9rem;">No applications yet.
              Share <a href="/workforpyx.php" style="color:var(--accent)">/workforpyx.php</a>.</p>
          <?php else: ?>
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Name</th>
                  <th>Track</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <?php foreach ($applications as $row): ?>
                  <?php
                    $rid = (string) ($row['id'] ?? '');
                    $active = $rid === $view_id;
                    $st = (string) ($row['status'] ?? 'new');
                    $pill = workforpyx_status_pill_class($st);
                    $when = (string) ($row['created'] ?? '');
                    if ($when !== '') {
                        $when = gmdate('Y-m-d H:i', strtotime($when)) . ' UTC';
                    }
                    $href = '/dev-workshop.php?id=' . rawurlencode($rid);
                  ?>
                  <tr class="<?php echo $active ? 'is-active' : ''; ?>">
                    <td><a class="row-link" href="<?php echo $href; ?>"><?php echo workforpyx_escape($when); ?></a></td>
                    <td><a class="row-link" href="<?php echo $href; ?>"><?php echo workforpyx_escape($row['name'] ?? ''); ?></a></td>
                    <td><a class="row-link" href="<?php echo $href; ?>"><?php echo workforpyx_escape($row['track_label'] ?? $row['track'] ?? ''); ?></a></td>
                    <td><a class="row-link" href="<?php echo $href; ?>"><span class="pill <?php echo $pill; ?>"><?php
                      echo workforpyx_escape(workforpyx_status_label($st));
                    ?></span></a></td>
                  </tr>
                <?php endforeach; ?>
              </tbody>
            </table>
          <?php endif; ?>
        </div>

        <div class="panel">
          <?php if (!$detail): ?>
            <h2>Application detail</h2>
            <p style="color:var(--muted);">Select an application from the list to read it and add an internal reply.</p>
          <?php else: ?>
            <h2><?php echo workforpyx_escape($detail['name'] ?? 'Applicant'); ?></h2>
            <p style="margin:0 0 12px;font-size:0.85rem;color:var(--muted);">
              <code><?php echo workforpyx_escape($detail['id'] ?? ''); ?></code>
              · <?php echo workforpyx_escape($detail['track_label'] ?? ''); ?>
            </p>

            <dl class="field">
              <dt>Email</dt>
              <dd><a href="mailto:<?php echo workforpyx_escape($detail['email'] ?? ''); ?>" style="color:var(--accent)"><?php
                echo workforpyx_escape($detail['email'] ?? '');
              ?></a></dd>
              <?php if (!empty($detail['phone'])): ?>
                <dt>Phone</dt>
                <dd><?php echo workforpyx_escape($detail['phone']); ?></dd>
              <?php endif; ?>
              <?php if (!empty($detail['location'])): ?>
                <dt>Location</dt>
                <dd><?php echo workforpyx_escape($detail['location']); ?></dd>
              <?php endif; ?>
              <?php if (!empty($detail['availability'])): ?>
                <dt>Availability</dt>
                <dd><?php echo workforpyx_escape($detail['availability']); ?></dd>
              <?php endif; ?>
              <?php if (!empty($detail['portfolio_url'])): ?>
                <dt>Portfolio</dt>
                <dd><a href="<?php echo workforpyx_escape($detail['portfolio_url']); ?>" target="_blank" rel="noopener" style="color:var(--accent)"><?php
                  echo workforpyx_escape($detail['portfolio_url']);
                ?></a></dd>
              <?php endif; ?>
              <dt>Resume</dt>
              <dd>
                <?php echo workforpyx_escape($detail['resume_original'] ?? 'file'); ?>
                · <a href="/dev-workshop.php?action=resume&amp;id=<?php echo urlencode((string) $detail['id']); ?>" style="color:var(--accent)">Download</a>
              </dd>
              <dt>Experience</dt>
              <dd><?php echo workforpyx_escape($detail['experience'] ?? '—'); ?></dd>
              <dt>Skills</dt>
              <dd><?php echo workforpyx_escape($detail['skills'] ?? '—'); ?></dd>
              <dt>Why Pyx</dt>
              <dd><?php echo workforpyx_escape($detail['why_pyx'] ?? '—'); ?></dd>
              <?php if (!empty($detail['message'])): ?>
                <dt>Extra notes</dt>
                <dd><?php echo workforpyx_escape($detail['message']); ?></dd>
              <?php endif; ?>
            </dl>

            <div class="decision-box">
              <h2>Decision email</h2>
              <p style="font-size:0.85rem;color:var(--muted);margin:0 0 10px;">
                Sends a branded HTML email to <strong><?php echo workforpyx_escape($detail['email'] ?? ''); ?></strong>
                from your configured Pyx SMTP address.
              </p>
              <?php if (!empty($detail['decision_emailed_at'])): ?>
                <p style="font-size:0.82rem;color:#6ee7b7;margin:0 0 10px;">
                  Last emailed: <?php echo workforpyx_escape((string) $detail['decision_emailed_at']); ?>
                  · Status: <strong><?php echo workforpyx_escape(workforpyx_status_label((string) ($detail['status'] ?? ''))); ?></strong>
                </p>
              <?php endif; ?>
              <form method="post" action="/dev-workshop.php?id=<?php echo urlencode((string) $detail['id']); ?>"
                onsubmit="return confirm('Send decision email to this applicant?');">
                <input type="hidden" name="action" value="decision">
                <input type="hidden" name="id" value="<?php echo workforpyx_escape((string) $detail['id']); ?>">
                <label for="decision_note" style="font-size:0.85rem;font-weight:600;">Optional note in email</label>
                <textarea id="decision_note" name="decision_note" rows="3" placeholder="Personal message (optional)…"><?php
                  echo workforpyx_escape((string) ($detail['decision_note'] ?? ''));
                ?></textarea>
                <div class="decision-actions">
                  <button type="submit" name="status" value="hired" class="btn btn-hired">Mark as hired</button>
                  <button type="submit" name="status" value="rejected" class="btn btn-rejected">Mark as rejected</button>
                </div>
              </form>
            </div>

            <div class="replies">
              <h2 style="margin-top:0;">Replies (internal)</h2>
              <p style="font-size:0.82rem;color:var(--muted);margin:0 0 10px;">
                Internal notes only — decision emails go out via the buttons above.
              </p>
              <?php
              $replies = $detail['replies'] ?? [];
              if (!$replies) {
                  echo '<p style="color:var(--muted);font-size:0.88rem;">No replies yet.</p>';
              } else {
                  foreach ($replies as $r) {
                      echo '<div class="reply"><meta>' . workforpyx_escape((string) ($r['at'] ?? '')) .
                          ' · ' . workforpyx_escape((string) ($r['from'] ?? 'dev')) . '</meta>' .
                          workforpyx_escape((string) ($r['body'] ?? '')) . '</div>';
                  }
              }
              ?>
              <form method="post" action="/dev-workshop.php?id=<?php echo urlencode((string) $detail['id']); ?>">
                <input type="hidden" name="action" value="reply">
                <input type="hidden" name="id" value="<?php echo workforpyx_escape((string) $detail['id']); ?>">
                <label for="reply_body" style="font-size:0.85rem;font-weight:600;">Add reply note</label>
                <textarea id="reply_body" name="reply_body" required placeholder="What you told the applicant (or plan to say)…"></textarea>
                <button type="submit" class="btn btn-primary">Save reply</button>
              </form>
            </div>
          <?php endif; ?>
        </div>
      </div>

      <?php endif; ?>
    </div>
  </div>

  <script src="/js/dev-workshop-traffic.js"></script>
  <script>
    (function () {
      var SESSION_KEY = "pyx_trainer_pw_ok";
      var gate = document.getElementById("gate");
      var workshop = document.getElementById("workshop");
      function showWorkshop() {
        gate.hidden = true;
        workshop.hidden = false;
        if (window.PyxDevWorkshopTraffic && /[?&]tab=traffic/.test(location.search)) {
          PyxDevWorkshopTraffic.init();
        }
      }
      try {
        if (sessionStorage.getItem(SESSION_KEY) === "1") {
          showWorkshop();
        } else {
          location.replace("/pyx-trainer-auth.html?next=" + encodeURIComponent(location.pathname + location.search));
        }
      } catch (e) {
        location.replace("/pyx-trainer-auth.html");
      }
      document.getElementById("lockWorkshop").addEventListener("click", function () {
        try { sessionStorage.removeItem(SESSION_KEY); } catch (e) {}
        location.href = "/pyx-trainer-auth.html";
      });
    })();
  </script>
</body>
</html>
