<?php
/**
 * Pyx — Work with Pyx / Dev Workshop shared storage (PHP).
 */
declare(strict_types=1);

function workforpyx_data_dir(): string
{
    $dir = dirname(__DIR__) . '/data/workforpyx';
    if (!is_dir($dir)) {
        mkdir($dir, 0755, true);
    }
    $resumes = $dir . '/resumes';
    if (!is_dir($resumes)) {
        mkdir($resumes, 0755, true);
    }
    return $dir;
}

function workforpyx_applications_path(): string
{
    return workforpyx_data_dir() . '/applications.json';
}

function workforpyx_load_applications(): array
{
    $path = workforpyx_applications_path();
    if (!is_file($path)) {
        return [];
    }
    $raw = file_get_contents($path);
    if ($raw === false || trim($raw) === '') {
        return [];
    }
    $data = json_decode($raw, true);
    return is_array($data) ? $data : [];
}

function workforpyx_save_applications(array $apps): bool
{
    $path = workforpyx_applications_path();
    $json = json_encode($apps, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
    if ($json === false) {
        return false;
    }
    return file_put_contents($path, $json . "\n", LOCK_EX) !== false;
}

function workforpyx_new_id(): string
{
    return 'app_' . gmdate('Ymd_His') . '_' . bin2hex(random_bytes(4));
}

function workforpyx_tracks(): array
{
    return [
        'ai_training' => [
            'label' => 'AI training & safety',
            'hint' => 'Help Pyx learn — feedback, labeling, moderation examples, and quality review.',
        ],
        'coding' => [
            'label' => 'Engineering & coding',
            'hint' => 'Build features, APIs, Studio apps, integrations, and developer tools.',
        ],
        'content' => [
            'label' => 'Writing & education',
            'hint' => 'Essays, docs, classroom copy, and kid-friendly explanations for Pyx Studio.',
        ],
        'design' => [
            'label' => 'Design & creative',
            'hint' => 'UI/UX, branding, pixel art direction, and visual polish across the web apps.',
        ],
        'community' => [
            'label' => 'Community & support',
            'hint' => 'Help users, forums, onboarding, and keeping Pyx friendly for everyone.',
        ],
        'operations' => [
            'label' => 'Operations & deploy',
            'hint' => 'Hosting, releases, monitoring, and keeping pyx-ai.web.app running smoothly.',
        ],
    ];
}

function workforpyx_escape(?string $s): string
{
    return htmlspecialchars((string) $s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function workforpyx_find_by_id(string $id): ?array
{
    foreach (workforpyx_load_applications() as $app) {
        if (($app['id'] ?? '') === $id) {
            return $app;
        }
    }
    return null;
}

function workforpyx_add_reply(string $id, string $body, string $from = 'dev'): bool
{
    $body = trim($body);
    if ($body === '') {
        return false;
    }
    $apps = workforpyx_load_applications();
    $found = false;
    foreach ($apps as &$app) {
        if (($app['id'] ?? '') !== $id) {
            continue;
        }
        if (!isset($app['replies']) || !is_array($app['replies'])) {
            $app['replies'] = [];
        }
        $app['replies'][] = [
            'at' => gmdate('c'),
            'from' => $from,
            'body' => $body,
        ];
        if (!in_array($app['status'] ?? '', ['hired', 'rejected'], true)) {
            $app['status'] = 'replied';
        }
        $found = true;
        break;
    }
    unset($app);
    return $found && workforpyx_save_applications($apps);
}

function workforpyx_update_status(string $id, string $status, string $note = ''): bool
{
    if (!in_array($status, ['hired', 'rejected'], true)) {
        return false;
    }
    $apps = workforpyx_load_applications();
    $found = false;
    foreach ($apps as &$app) {
        if (($app['id'] ?? '') !== $id) {
            continue;
        }
        $app['status'] = $status;
        $app['decision_at'] = gmdate('c');
        $app['decision_note'] = $note;
        $found = true;
        break;
    }
    unset($app);
    return $found && workforpyx_save_applications($apps);
}

function workforpyx_allowed_resume_ext(string $name): bool
{
    $ext = strtolower(pathinfo($name, PATHINFO_EXTENSION));
    return in_array($ext, ['pdf', 'doc', 'docx', 'txt', 'rtf'], true);
}

function workforpyx_move_resume_file(string $tmp, string $dest): bool
{
    if ($tmp === '' || !is_file($tmp)) {
        return false;
    }
    if (is_uploaded_file($tmp)) {
        return move_uploaded_file($tmp, $dest);
    }
    // Cloud Run Flask→PHP bridge uses temp files, not HTTP uploads.
    if (@rename($tmp, $dest)) {
        return true;
    }
    return @copy($tmp, $dest);
}

function workforpyx_store_resume(array $file, string $app_id): ?string
{
    if (($file['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_OK) {
        return null;
    }
    $orig = basename((string) ($file['name'] ?? ''));
    if ($orig === '' || !workforpyx_allowed_resume_ext($orig)) {
        return null;
    }
    $tmp = (string) ($file['tmp_name'] ?? '');
    $size = (int) ($file['size'] ?? 0);
    if ($size <= 0 && is_file($tmp)) {
        $size = (int) filesize($tmp);
    }
    if ($size <= 0 || $size > 30 * 1024 * 1024) {
        return null;
    }
    $ext = strtolower(pathinfo($orig, PATHINFO_EXTENSION));
    $safe = preg_replace('/[^a-zA-Z0-9._-]+/', '_', pathinfo($orig, PATHINFO_FILENAME));
    $stored = $app_id . '_' . ($safe ?: 'resume') . '.' . $ext;
    $dest = workforpyx_data_dir() . '/resumes/' . $stored;
    if (!workforpyx_move_resume_file($tmp, $dest)) {
        return null;
    }
    return $stored;
}
