<?php
/**
 * Work with Pyx — job / contributor applications (resume required).
 */
declare(strict_types=1);

require_once __DIR__ . '/workforpyx_lib.php';

$tracks = workforpyx_tracks();
$errors = [];
$success = false;
$submitted_id = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $track = (string) ($_POST['track'] ?? '');
    $name = trim((string) ($_POST['name'] ?? ''));
    $email = trim((string) ($_POST['email'] ?? ''));
    $phone = trim((string) ($_POST['phone'] ?? ''));
    $location = trim((string) ($_POST['location'] ?? ''));
    $experience = trim((string) ($_POST['experience'] ?? ''));
    $why = trim((string) ($_POST['why_pyx'] ?? ''));
    $skills = trim((string) ($_POST['skills'] ?? ''));
    $availability = trim((string) ($_POST['availability'] ?? ''));
    $portfolio = trim((string) ($_POST['portfolio_url'] ?? ''));
    $message = trim((string) ($_POST['message'] ?? ''));

    if (!isset($tracks[$track])) {
        $errors[] = 'Please choose what you want to work on.';
    }
    if ($name === '') {
        $errors[] = 'Please enter your name.';
    }
    if ($email === '' || !filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $errors[] = 'Please enter a valid email address.';
    }
    if ($why === '') {
        $errors[] = 'Tell us why you want to work with Pyx.';
    }
    if (!isset($_FILES['resume']) || ($_FILES['resume']['error'] ?? UPLOAD_ERR_NO_FILE) === UPLOAD_ERR_NO_FILE) {
        $errors[] = 'A resume file is required (PDF, Word, or text).';
    }

    if (!$errors) {
        $id = workforpyx_new_id();
        $resume_stored = workforpyx_store_resume($_FILES['resume'], $id);
        if (!$resume_stored) {
            $errors[] = 'Could not save your resume. Use PDF, DOC, DOCX, TXT, or RTF (file too large or wrong type).';
        } else {
            $apps = workforpyx_load_applications();
            $apps[] = [
                'id' => $id,
                'created' => gmdate('c'),
                'track' => $track,
                'track_label' => $tracks[$track]['label'],
                'name' => $name,
                'email' => $email,
                'phone' => $phone,
                'location' => $location,
                'experience' => $experience,
                'why_pyx' => $why,
                'skills' => $skills,
                'availability' => $availability,
                'portfolio_url' => $portfolio,
                'message' => $message,
                'resume_original' => (string) ($_FILES['resume']['name'] ?? ''),
                'resume_stored' => $resume_stored,
                'status' => 'new',
                'replies' => [],
            ];
            if (workforpyx_save_applications($apps)) {
                $success = true;
                $submitted_id = $id;
            } else {
                $errors[] = 'Could not save your application. Try again in a moment.';
            }
        }
    }
}

$form = $_POST;
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Work with Pyx — Apply</title>
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
      --accent2: #818cf8;
      --ok: #34d399;
      --err: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Plus Jakarta Sans", system-ui, sans-serif;
      background: linear-gradient(160deg, #0f172a 0%, #0c4a6e 45%, #1e1b4b 100%);
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
    }
    .wrap { max-width: 720px; margin: 0 auto; padding: 28px 18px 48px; }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(1.6rem, 4vw, 2rem);
      background: linear-gradient(135deg, #a5f3fc, #c4b5fd);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .lede { color: var(--muted); margin: 0 0 24px; max-width: 58ch; }
    .card {
      background: rgba(30, 41, 59, 0.85);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 16px;
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; }
    .hint { font-size: 0.8rem; color: var(--muted); font-weight: 400; margin: -2px 0 8px; }
    input, textarea, select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font: inherit;
      margin-bottom: 14px;
    }
    textarea { min-height: 88px; resize: vertical; }
    .tracks { display: grid; gap: 10px; margin-bottom: 16px; }
    .track {
      display: block;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #0f172a;
      cursor: pointer;
      transition: border-color 0.15s;
    }
    .track:has(input:checked) { border-color: var(--accent); background: rgba(56, 189, 248, 0.08); }
    .track input { width: auto; margin: 0 10px 0 0; vertical-align: middle; }
    .track strong { display: inline; font-size: 0.95rem; }
    .track span { display: block; margin-top: 4px; margin-left: 28px; font-size: 0.82rem; color: var(--muted); }
    .btn {
      display: inline-block;
      padding: 12px 22px;
      border: none;
      border-radius: 10px;
      background: linear-gradient(135deg, #0ea5e9, #6366f1);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .btn:hover { filter: brightness(1.08); }
    .errors {
      background: rgba(248, 113, 113, 0.12);
      border: 1px solid rgba(248, 113, 113, 0.4);
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 16px;
      color: #fecaca;
      font-size: 0.9rem;
    }
    .errors ul { margin: 0; padding-left: 18px; }
    .success {
      background: rgba(52, 211, 153, 0.12);
      border: 1px solid rgba(52, 211, 153, 0.4);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 20px;
    }
    .success h2 { margin: 0 0 8px; color: var(--ok); font-size: 1.1rem; }
    .foot { margin-top: 24px; font-size: 0.85rem; color: var(--muted); }
    .foot a { color: var(--accent2); }
    .req::after { content: " *"; color: var(--accent); }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Work with Pyx</h1>
    <p class="lede">
      Help build Pyx Studio — train the AI, write code, shape content, and more.
      Every application needs a <strong>resume file</strong> so we can learn about you.
    </p>

    <?php if ($success): ?>
      <div class="success">
        <h2>Application received</h2>
        <p>Thanks, <?php echo workforpyx_escape($form['name'] ?? 'there'); ?>! We saved your application
          <code><?php echo workforpyx_escape($submitted_id); ?></code>.
          If we move forward, we will email you at the address you provided.</p>
        <p><a href="/workforpyx.php" style="color:var(--accent)">Submit another application</a>
          · <a href="/" style="color:var(--accent2)">Pyx Studio home</a></p>
      </div>
    <?php else: ?>

      <?php if ($errors): ?>
        <div class="errors" role="alert">
          <ul>
            <?php foreach ($errors as $e): ?>
              <li><?php echo workforpyx_escape($e); ?></li>
            <?php endforeach; ?>
          </ul>
        </div>
      <?php endif; ?>

      <form method="post" action="/workforpyx.php" enctype="multipart/form-data">
        <div class="card">
          <h2>What do you want to do?</h2>
          <p class="hint">Pick the area that fits you best. You can only submit one track per application.</p>
          <div class="tracks">
            <?php foreach ($tracks as $key => $meta): ?>
              <label class="track">
                <input type="radio" name="track" value="<?php echo workforpyx_escape($key); ?>" required
                  <?php echo (($form['track'] ?? '') === $key) ? 'checked' : ''; ?>>
                <strong><?php echo workforpyx_escape($meta['label']); ?></strong>
                <span><?php echo workforpyx_escape($meta['hint']); ?></span>
              </label>
            <?php endforeach; ?>
          </div>
        </div>

        <div class="card">
          <h2>About you</h2>
          <label class="req" for="name">Full name</label>
          <input id="name" name="name" required maxlength="120"
            value="<?php echo workforpyx_escape($form['name'] ?? ''); ?>">

          <label class="req" for="email">Email</label>
          <input id="email" name="email" type="email" required maxlength="200"
            value="<?php echo workforpyx_escape($form['email'] ?? ''); ?>">

          <label for="phone">Phone (optional)</label>
          <input id="phone" name="phone" type="tel" maxlength="40"
            value="<?php echo workforpyx_escape($form['phone'] ?? ''); ?>">

          <label for="location">City / time zone (optional)</label>
          <input id="location" name="location" maxlength="120"
            value="<?php echo workforpyx_escape($form['location'] ?? ''); ?>">

          <label class="req" for="resume">Resume file</label>
          <p class="hint">PDF, Word, or text. Required for all roles.</p>
          <input id="resume" name="resume" type="file" accept=".pdf,.doc,.docx,.txt,.rtf" required>
        </div>

        <div class="card">
          <h2>Your experience</h2>
          <label for="experience">Work, school, or projects</label>
          <textarea id="experience" name="experience" rows="4" maxlength="4000"
            placeholder="What have you built, studied, or contributed to before?"><?php
            echo workforpyx_escape($form['experience'] ?? '');
          ?></textarea>

          <label for="skills">Skills & tools</label>
          <textarea id="skills" name="skills" rows="3" maxlength="2000"
            placeholder="Languages, frameworks, writing, design tools, etc."><?php
            echo workforpyx_escape($form['skills'] ?? '');
          ?></textarea>

          <label for="portfolio_url">Portfolio or GitHub (optional)</label>
          <input id="portfolio_url" name="portfolio_url" type="url" maxlength="500"
            placeholder="https://"
            value="<?php echo workforpyx_escape($form['portfolio_url'] ?? ''); ?>">

          <label for="availability">Availability</label>
          <input id="availability" name="availability" maxlength="200"
            placeholder="e.g. 5 hrs/week, summer only, flexible"
            value="<?php echo workforpyx_escape($form['availability'] ?? ''); ?>">
        </div>

        <div class="card">
          <h2>Why Pyx?</h2>
          <label class="req" for="why_pyx">Why do you want to work with Pyx?</label>
          <textarea id="why_pyx" name="why_pyx" rows="4" required maxlength="3000"><?php
            echo workforpyx_escape($form['why_pyx'] ?? '');
          ?></textarea>

          <label for="message">Anything else? (optional)</label>
          <textarea id="message" name="message" rows="3" maxlength="2000"><?php
            echo workforpyx_escape($form['message'] ?? '');
          ?></textarea>
        </div>

        <button type="submit" class="btn">Submit application</button>
      </form>
    <?php endif; ?>

    <p class="foot">
      <a href="/">← Pyx Studio</a>
      · Staff: <a href="/dev-workshop.php">Dev Workshop</a> (password)
    </p>
  </div>
</body>
</html>
