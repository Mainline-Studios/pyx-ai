<?php
/**
 * Cloud Run / Flask bridge so CLI PHP can serve web requests.
 * Set PYX_PHP_BRIDGE=1 in the environment.
 */
declare(strict_types=1);

if (getenv('PYX_PHP_BRIDGE') !== '1') {
    return;
}

parse_str((string) getenv('QUERY_STRING'), $_GET);
$_SERVER['REQUEST_METHOD'] = getenv('REQUEST_METHOD') ?: 'GET';
$_SERVER['REQUEST_URI'] = getenv('REQUEST_URI') ?: '';

$post_json = getenv('PYX_POST_JSON');
if ($post_json !== false && $post_json !== '') {
    $decoded = json_decode($post_json, true);
    $_POST = is_array($decoded) ? $decoded : [];
}

$files_json = getenv('PYX_FILES_JSON');
if ($files_json !== false && $files_json !== '') {
    $spec = json_decode($files_json, true);
    $_FILES = [];
    if (is_array($spec)) {
        foreach ($spec as $field => $info) {
            if (!is_array($info) || empty($info['path']) || !is_file($info['path'])) {
                continue;
            }
            $_FILES[$field] = [
                'name' => $info['name'] ?? 'upload',
                'type' => $info['type'] ?? 'application/octet-stream',
                'tmp_name' => $info['path'],
                'error' => UPLOAD_ERR_OK,
                'size' => (int) ($info['size'] ?? filesize($info['path'])),
            ];
        }
    }
}
