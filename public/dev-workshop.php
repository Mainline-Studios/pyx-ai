<?php
/**
 * Pyx Dev Workshop — review applications & reply (staff; same password as trainer gate).
 */
declare(strict_types=1);

require_once __DIR__ . '/workforpyx_lib.php';

$tracks = workforpyx_tracks();
$flash = '';
$view_id = trim((string) ($_GET['id'] ?? ''));
$action = (string) ($_GET['action'] ?? '');

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
        'replied' => 'Replied',
        'reviewing' => 'Reviewing',
        'new' => 'New',
        default => ucfirst($status),
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

      <div class="features" aria-label="Workshop features">
        <span class="feature is-on">Applications inbox</span>
        <span class="feature is-on">Reply notes</span>
        <span class="feature is-soon">More coming soon</span>
      </div>

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
                    $pill = $st === 'replied' ? 'pill--replied' : 'pill--new';
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

            <div class="replies">
              <h2 style="margin-top:0;">Replies (internal)</h2>
              <p style="font-size:0.82rem;color:var(--muted);margin:0 0 10px;">
                Saved here for your team. Copy your reply and email the applicant manually.
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
    </div>
  </div>

  <script>
    (function () {
      var SESSION_KEY = "pyx_trainer_pw_ok";
      var gate = document.getElementById("gate");
      var workshop = document.getElementById("workshop");
      function showWorkshop() {
        gate.hidden = true;
        workshop.hidden = false;
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
