<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
  <title>MARII Ask API :: How to use it in your code</title>
</head>
<body bgcolor="#c0c0c0" text="#000000" link="#0000ee" vlink="#551a8b" alink="#ff0000">
<font face="Verdana, Arial, Helvetica, sans-serif" size="2">

<table width="720" border="0" cellpadding="0" cellspacing="0" bgcolor="#000080">
  <tr>
    <td>
      <table width="100%" border="0" cellpadding="8" cellspacing="0">
        <tr>
          <td><font face="Verdana, Arial, Helvetica, sans-serif" size="4" color="#ffffff"><b>MARII ASK API</b></font><br>
          <font face="Verdana, Arial, Helvetica, sans-serif" size="1" color="#ffff00">how to call it from your code &nbsp;|&nbsp; last updated oct 2026</font></td>
        </tr>
      </table>
    </td>
  </tr>
</table>

<table width="720" border="1" cellpadding="12" cellspacing="0" bgcolor="#ffffff" bordercolor="#808080">
  <tr>
    <td>

<p><font size="2">This is a <b>GET</b> JSON API. You put the question in the URL. No key. No POST body. No cloud LLM.</font></p>

<p><font size="2" color="#000080"><b>1. The URL</b></font><br>
<font size="2">Base:</font><br>
<font face="Courier New, Courier, monospace" size="2">https://marii-ask.mainline-mi.workers.dev/</font><br>
<font size="2">Stick your question on the end. Spaces become <b>%20</b>.</font></p>

<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#ffffcc" bordercolor="#cccc99">
  <tr>
    <td><font face="Courier New, Courier, monospace" size="2">
https://marii-ask.mainline-mi.workers.dev/how%20is%20judge<br>
https://marii-ask.mainline-mi.workers.dev/what%20is%20popcorn<br>
https://marii-ask.mainline-mi.workers.dev/15%25%20of%2080
    </font></td>
  </tr>
</table>

<p><font size="2">Same thing via the site (follow redirects):</font><br>
<font face="Courier New, Courier, monospace" size="2">https://pyx-ai.web.app/api/marii/ask/how%20is%20judge</font></p>

<p><font size="2" color="#000080"><b>2. What you get back</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#f4f4f4" bordercolor="#cccccc">
  <tr>
    <td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">{
  "ok": true,
  "request": "how is judge",
  "reply": "Aaron Judge (#99) ...",
  "echo": "how is judge",
  "source": "sports",
  "backend": "pyx-assistant",
  "ai": false
}</pre></td>
  </tr>
</table>
<p><font size="2">Read <b>reply</b>. <b>source</b> is sports / local / wiki / weather / kb-miss.</font></p>

<p><font size="2" color="#000080"><b>3. PHP (old and simple)</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99">
  <tr>
    <td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">&lt;?php
$q = "how is judge";
$url = "https://marii-ask.mainline-mi.workers.dev/" . rawurlencode($q);

$json = file_get_contents($url);
$data = json_decode($json, true);

echo $data["reply"];
?&gt;</pre></td>
  </tr>
</table>

<p><font size="2" color="#000080"><b>4. PHP with curl</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99">
  <tr>
    <td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">&lt;?php
$q = "how is judge";
$url = "https://marii-ask.mainline-mi.workers.dev/" . rawurlencode($q);

$ch = curl_init($url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
$json = curl_exec($ch);
curl_close($ch);

$data = json_decode($json, true);
if (!empty($data["ok"])) {
  echo $data["reply"];
}
?&gt;</pre></td>
  </tr>
</table>

<p><font size="2" color="#000080"><b>5. Shell</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeeeff" bordercolor="#9999cc">
  <tr>
    <td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">curl -sS "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"</pre></td>
  </tr>
</table>

<p><font size="2" color="#000080"><b>6. JavaScript</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#ffeeee" bordercolor="#cc9999">
  <tr>
    <td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">fetch("https://marii-ask.mainline-mi.workers.dev/" + encodeURIComponent("how is judge"))
  .then(function (r) { return r.json(); })
  .then(function (data) { console.log(data.reply); });</pre></td>
  </tr>
</table>

<hr size="1" noshade color="#808080">
<p><font size="1" color="#333333">
GET only. POST is 501 on purpose (no cloud AI).<br>
Encode the question: space = %20 &nbsp; percent = %25<br>
<b>file_get_contents</b> needs allow_url_fopen. If that is off, use the curl block.<br>
<a href="/mainlineintelligence#marii-ask">back to mainline intelligence</a>
&nbsp;|&nbsp;
<a href="https://marii-ask.mainline-mi.workers.dev/health">api health</a>
</font></p>

    </td>
  </tr>
</table>

<p><font face="Verdana, Arial, Helvetica, sans-serif" size="1" color="#333333">mainline intelligence &nbsp;--&nbsp; marii ask &nbsp;--&nbsp; copy the snippet into your app</font></p>

</font>
</body>
</html>
