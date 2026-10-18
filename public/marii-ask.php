<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
  <title>MARII Ask API :: How to use it in every language</title>
</head>
<body bgcolor="#c0c0c0" text="#000000" link="#0000ee" vlink="#551a8b" alink="#ff0000">
<font face="Verdana, Arial, Helvetica, sans-serif" size="2">

<table width="760" border="0" cellpadding="0" cellspacing="0" bgcolor="#000080">
  <tr>
    <td>
      <table width="100%" border="0" cellpadding="8" cellspacing="0">
        <tr>
          <td><font face="Verdana, Arial, Helvetica, sans-serif" size="4" color="#ffffff"><b>MARII ASK API</b></font><br>
          <font face="Verdana, Arial, Helvetica, sans-serif" size="1" color="#ffff00">how to call it from your code &nbsp;|&nbsp; one GET, many languages &nbsp;|&nbsp; oct 2026</font></td>
        </tr>
      </table>
    </td>
  </tr>
</table>

<table width="760" border="1" cellpadding="12" cellspacing="0" bgcolor="#ffffff" bordercolor="#808080">
  <tr>
    <td>

<p><font size="2">This is a <b>GET</b> JSON API. Put the question in the URL. No key. No POST. No cloud LLM. Copy a snippet into <b>your</b> program.</font></p>

<p><font size="2" color="#000080"><b>The URL</b></font><br>
<font face="Courier New, Courier, monospace" size="2">https://marii-ask.mainline-mi.workers.dev/</font>
<font size="2"> + url-encoded question. Space = <b>%20</b>. Percent = <b>%25</b>.</font></p>

<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#ffffcc" bordercolor="#cccc99">
  <tr>
    <td><font face="Courier New, Courier, monospace" size="2">
https://marii-ask.mainline-mi.workers.dev/how%20is%20judge<br>
https://marii-ask.mainline-mi.workers.dev/what%20is%20popcorn<br>
https://marii-ask.mainline-mi.workers.dev/15%25%20of%2080<br>
https://pyx-ai.web.app/api/marii/ask/how%20is%20judge
    </font></td>
  </tr>
</table>

<p><font size="2" color="#000080"><b>What you get back</b></font> &nbsp; <font size="2">read <b>reply</b></font></p>
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

<p><font size="2" color="#000080"><b>Jump</b></font><br>
<font size="1">
<a href="#curl">curl</a> &middot; <a href="#wget">wget</a> &middot; <a href="#http">HTTP</a> &middot;
<a href="#php">PHP</a> &middot; <a href="#python">Python</a> &middot; <a href="#js">JavaScript</a> &middot;
<a href="#ts">TypeScript</a> &middot; <a href="#node">Node</a> &middot; <a href="#deno">Deno</a> &middot;
<a href="#ruby">Ruby</a> &middot; <a href="#perl">Perl</a> &middot; <a href="#go">Go</a> &middot;
<a href="#rust">Rust</a> &middot; <a href="#java">Java</a> &middot; <a href="#kotlin">Kotlin</a> &middot;
<a href="#csharp">C#</a> &middot; <a href="#fsharp">F#</a> &middot; <a href="#vb">VB.NET</a> &middot;
<a href="#c">C</a> &middot; <a href="#cpp">C++</a> &middot; <a href="#objc">Obj-C</a> &middot;
<a href="#swift">Swift</a> &middot; <a href="#dart">Dart</a> &middot; <a href="#r">R</a> &middot;
<a href="#lua">Lua</a> &middot; <a href="#ps">PowerShell</a> &middot; <a href="#bat">Batch</a> &middot;
<a href="#as">AppleScript</a> &middot; <a href="#scala">Scala</a> &middot; <a href="#groovy">Groovy</a> &middot;
<a href="#clj">Clojure</a> &middot; <a href="#ex">Elixir</a> &middot; <a href="#erl">Erlang</a> &middot;
<a href="#hs">Haskell</a> &middot; <a href="#ml">OCaml</a> &middot; <a href="#jl">Julia</a> &middot;
<a href="#mat">MATLAB</a> &middot; <a href="#rkt">Racket</a> &middot; <a href="#cr">Crystal</a> &middot;
<a href="#nim">Nim</a> &middot; <a href="#zig">Zig</a> &middot; <a href="#d">D</a> &middot;
<a href="#pas">Pascal</a> &middot; <a href="#f90">Fortran</a> &middot; <a href="#cob">COBOL</a> &middot;
<a href="#ada">Ada</a> &middot; <a href="#tcl">Tcl</a> &middot; <a href="#awk">Awk</a> &middot;
<a href="#vba">VBA</a> &middot; <a href="#ahk">AutoHotkey</a> &middot; <a href="#html">HTML</a> &middot;
<a href="#cfml">CFML</a> &middot; <a href="#plg">Prolog</a> &middot; <a href="#lisp">Common Lisp</a>
</font></p>

<hr size="1" noshade color="#808080">

<p><a name="curl"></a><font size="2" color="#000080"><b>curl / sh</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeeeff" bordercolor="#9999cc"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">curl -sS "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"</pre></td></tr></table>

<p><a name="wget"></a><font size="2" color="#000080"><b>wget</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeeeff" bordercolor="#9999cc"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">wget -qO- "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"</pre></td></tr></table>

<p><a name="http"></a><font size="2" color="#000080"><b>raw HTTP</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeeeff" bordercolor="#9999cc"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">GET /how%20is%20judge HTTP/1.1
Host: marii-ask.mainline-mi.workers.dev
Accept: application/json</pre></td></tr></table>

<p><a name="php"></a><font size="2" color="#000080"><b>PHP</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">&lt;?php
$q = "how is judge";
$url = "https://marii-ask.mainline-mi.workers.dev/" . rawurlencode($q);
$json = file_get_contents($url);
$data = json_decode($json, true);
echo $data["reply"];
?&gt;</pre></td></tr></table>
<p><font size="1">If allow_url_fopen is off:</font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">&lt;?php
$url = "https://marii-ask.mainline-mi.workers.dev/" . rawurlencode("how is judge");
$ch = curl_init($url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
$data = json_decode(curl_exec($ch), true);
curl_close($ch);
echo $data["reply"];
?&gt;</pre></td></tr></table>

<p><a name="python"></a><font size="2" color="#000080"><b>Python</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">from urllib.parse import quote
from urllib.request import urlopen
import json
url = "https://marii-ask.mainline-mi.workers.dev/" + quote("how is judge")
data = json.load(urlopen(url))
print(data["reply"])</pre></td></tr></table>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">import requests
r = requests.get("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge")
print(r.json()["reply"])</pre></td></tr></table>

<p><a name="js"></a><font size="2" color="#000080"><b>JavaScript (browser)</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#ffeeee" bordercolor="#cc9999"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">fetch("https://marii-ask.mainline-mi.workers.dev/" + encodeURIComponent("how is judge"))
  .then(function (r) { return r.json(); })
  .then(function (data) { console.log(data.reply); });</pre></td></tr></table>

<p><a name="ts"></a><font size="2" color="#000080"><b>TypeScript</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#ffeeee" bordercolor="#cc9999"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">const url = "https://marii-ask.mainline-mi.workers.dev/" + encodeURIComponent("how is judge");
const data = await (await fetch(url)).json();
console.log(data.reply as string);</pre></td></tr></table>

<p><a name="node"></a><font size="2" color="#000080"><b>Node.js</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#ffeeee" bordercolor="#cc9999"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">const url = "https://marii-ask.mainline-mi.workers.dev/" + encodeURIComponent("how is judge");
const data = await (await fetch(url)).json();
console.log(data.reply);</pre></td></tr></table>

<p><a name="deno"></a><font size="2" color="#000080"><b>Deno / Bun</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#ffeeee" bordercolor="#cc9999"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">const data = await (await fetch(
  "https://marii-ask.mainline-mi.workers.dev/" + encodeURIComponent("how is judge")
)).json();
console.log(data.reply);</pre></td></tr></table>

<p><a name="ruby"></a><font size="2" color="#000080"><b>Ruby</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">require "open-uri"
require "json"
require "uri"
url = "https://marii-ask.mainline-mi.workers.dev/" + URI.encode_www_form_component("how is judge")
puts JSON.parse(URI.open(url).read)["reply"]</pre></td></tr></table>

<p><a name="perl"></a><font size="2" color="#000080"><b>Perl</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">use LWP::Simple;
use JSON;
my $url = "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge";
print decode_json(get($url))-&gt;{reply}, "\n";</pre></td></tr></table>

<p><a name="go"></a><font size="2" color="#000080"><b>Go</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">package main
import (
  "encoding/json"
  "fmt"
  "net/http"
  "net/url"
)
func main() {
  u := "https://marii-ask.mainline-mi.workers.dev/" + url.PathEscape("how is judge")
  res, _ := http.Get(u)
  defer res.Body.Close()
  var data map[string]any
  json.NewDecoder(res.Body).Decode(&amp;data)
  fmt.Println(data["reply"])
}</pre></td></tr></table>

<p><a name="rust"></a><font size="2" color="#000080"><b>Rust</b></font> <font size="1">(reqwest + serde_json)</font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">let url = format!(
  "https://marii-ask.mainline-mi.workers.dev/{}",
  urlencoding::encode("how is judge")
);
let v: serde_json::Value = reqwest::blocking::get(url)?.json()?;
println!("{}", v["reply"]);</pre></td></tr></table>

<p><a name="java"></a><font size="2" color="#000080"><b>Java</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">import java.net.*;
import java.net.http.*;
String q = URLEncoder.encode("how is judge", "UTF-8");
String url = "https://marii-ask.mainline-mi.workers.dev/" + q;
String body = HttpClient.newHttpClient()
  .send(HttpRequest.newBuilder(URI.create(url)).GET().build(),
        HttpResponse.BodyHandlers.ofString()).body();
System.out.println(body);</pre></td></tr></table>

<p><a name="kotlin"></a><font size="2" color="#000080"><b>Kotlin</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">val q = java.net.URLEncoder.encode("how is judge", "UTF-8")
val text = java.net.URL("https://marii-ask.mainline-mi.workers.dev/$q").readText()
println(text)</pre></td></tr></table>

<p><a name="csharp"></a><font size="2" color="#000080"><b>C#</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">using var http = new HttpClient();
var q = Uri.EscapeDataString("how is judge");
var json = await http.GetStringAsync("https://marii-ask.mainline-mi.workers.dev/" + q);
Console.WriteLine(json);</pre></td></tr></table>

<p><a name="fsharp"></a><font size="2" color="#000080"><b>F#</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">open System.Net.Http
let http = new HttpClient()
let q = System.Uri.EscapeDataString("how is judge")
let json = http.GetStringAsync("https://marii-ask.mainline-mi.workers.dev/" + q).Result
printfn "%s" json</pre></td></tr></table>

<p><a name="vb"></a><font size="2" color="#000080"><b>VB.NET</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">Dim http As New System.Net.Http.HttpClient()
Dim q = Uri.EscapeDataString("how is judge")
Dim json = http.GetStringAsync("https://marii-ask.mainline-mi.workers.dev/" &amp; q).Result
Console.WriteLine(json)</pre></td></tr></table>

<p><a name="c"></a><font size="2" color="#000080"><b>C</b></font> <font size="1">(libcurl)</font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">#include &lt;curl/curl.h&gt;
CURL *c = curl_easy_init();
curl_easy_setopt(c, CURLOPT_URL,
  "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge");
curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
curl_easy_perform(c);
curl_easy_cleanup(c);</pre></td></tr></table>

<p><a name="cpp"></a><font size="2" color="#000080"><b>C++</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">// same as C with libcurl, or:
// #include &lt;cpr/cpr.h&gt;
auto r = cpr::Get(cpr::Url{
  "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"});
std::cout &lt;&lt; r.text;</pre></td></tr></table>

<p><a name="objc"></a><font size="2" color="#000080"><b>Objective-C</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">NSURL *u = [NSURL URLWithString:
  @"https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"];
NSData *d = [NSData dataWithContentsOfURL:u];
NSLog(@"%@", [[NSString alloc] initWithData:d encoding:NSUTF8StringEncoding]);</pre></td></tr></table>

<p><a name="swift"></a><font size="2" color="#000080"><b>Swift</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">let q = "how is judge".addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!
let url = URL(string: "https://marii-ask.mainline-mi.workers.dev/\(q)")!
let (data, _) = try await URLSession.shared.data(from: url)
print(String(data: data, encoding: .utf8)!)</pre></td></tr></table>

<p><a name="dart"></a><font size="2" color="#000080"><b>Dart</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">import 'dart:convert';
import 'package:http/http.dart' as http;
final q = Uri.encodeComponent("how is judge");
final r = await http.get(Uri.parse("https://marii-ask.mainline-mi.workers.dev/$q"));
print(jsonDecode(r.body)["reply"]);</pre></td></tr></table>

<p><a name="r"></a><font size="2" color="#000080"><b>R</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">j &lt;- jsonlite::fromJSON("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge")
cat(j$reply, "\n")</pre></td></tr></table>

<p><a name="lua"></a><font size="2" color="#000080"><b>Lua</b></font> <font size="1">(luacurl or http.request)</font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">local http = require("socket.http")
local body = http.request("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge")
print(body)</pre></td></tr></table>

<p><a name="ps"></a><font size="2" color="#000080"><b>PowerShell</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeeeff" bordercolor="#9999cc"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">$r = Invoke-RestMethod "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"
$r.reply</pre></td></tr></table>

<p><a name="bat"></a><font size="2" color="#000080"><b>Windows Batch</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeeeff" bordercolor="#9999cc"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">curl -sS "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"</pre></td></tr></table>

<p><a name="as"></a><font size="2" color="#000080"><b>AppleScript</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">do shell script "curl -sS 'https://marii-ask.mainline-mi.workers.dev/how%20is%20judge'"</pre></td></tr></table>

<p><a name="scala"></a><font size="2" color="#000080"><b>Scala</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">val src = scala.io.Source.fromURL(
  "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge")
println(src.mkString)
src.close()</pre></td></tr></table>

<p><a name="groovy"></a><font size="2" color="#000080"><b>Groovy</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">println "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge".toURL().text</pre></td></tr></table>

<p><a name="clj"></a><font size="2" color="#000080"><b>Clojure</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">(println (slurp "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"))</pre></td></tr></table>

<p><a name="ex"></a><font size="2" color="#000080"><b>Elixir</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">{:ok, {_, _, body}} = :httpc.request(
  'https://marii-ask.mainline-mi.workers.dev/how%20is%20judge')
IO.puts(body)</pre></td></tr></table>

<p><a name="erl"></a><font size="2" color="#000080"><b>Erlang</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">inets:start(),
ssl:start(),
{ok, {_, _, Body}} = httpc:request(get,
  {"https://marii-ask.mainline-mi.workers.dev/how%20is%20judge", []}, [], []),
io:format("~s~n", [Body]).</pre></td></tr></table>

<p><a name="hs"></a><font size="2" color="#000080"><b>Haskell</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">import Network.HTTP.Simple
r &lt;- httpBS "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"
print (getResponseBody r)</pre></td></tr></table>

<p><a name="ml"></a><font size="2" color="#000080"><b>OCaml</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">(* cohttp-lwt-unix *)
let%lwt _, body = Cohttp_lwt_unix.Client.get
  (Uri.of_string "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge") in
Cohttp_lwt.Body.to_string body |&gt; Lwt_io.print</pre></td></tr></table>

<p><a name="jl"></a><font size="2" color="#000080"><b>Julia</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">using HTTP, JSON
j = JSON.parse(String(HTTP.get("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge").body))
println(j["reply"])</pre></td></tr></table>

<p><a name="mat"></a><font size="2" color="#000080"><b>MATLAB / Octave</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">txt = webread("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge");
disp(txt.reply)</pre></td></tr></table>

<p><a name="rkt"></a><font size="2" color="#000080"><b>Racket</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">#lang racket
(require net/url)
(port-&gt;string (get-pure-port
  (string-&gt;url "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge")))</pre></td></tr></table>

<p><a name="cr"></a><font size="2" color="#000080"><b>Crystal</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">require "http/client"
puts HTTP::Client.get("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge").body</pre></td></tr></table>

<p><a name="nim"></a><font size="2" color="#000080"><b>Nim</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">import httpclient
echo newHttpClient().getContent("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge")</pre></td></tr></table>

<p><a name="zig"></a><font size="2" color="#000080"><b>Zig</b></font> <font size="1">(or just shell out)</font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">// easiest: spawn curl
// std.process.Child.run(.{ .argv = &.{ "curl", "-sS",
//   "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge" } })</pre></td></tr></table>

<p><a name="d"></a><font size="2" color="#000080"><b>D</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">import std.net.curl;
writeln(get("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"));</pre></td></tr></table>

<p><a name="pas"></a><font size="2" color="#000080"><b>Pascal / Delphi</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">// Indy
IdHTTP1.Get('https://marii-ask.mainline-mi.workers.dev/how%20is%20judge');</pre></td></tr></table>

<p><a name="f90"></a><font size="2" color="#000080"><b>Fortran</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">! call curl from the shell
call execute_command_line(
  "curl -sS https://marii-ask.mainline-mi.workers.dev/how%20is%20judge")</pre></td></tr></table>

<p><a name="cob"></a><font size="2" color="#000080"><b>COBOL</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">       IDENTIFICATION DIVISION.
       PROGRAM-ID. MARII.
       PROCEDURE DIVISION.
           CALL "SYSTEM" USING
             "curl -sS https://marii-ask.mainline-mi.workers.dev/how%20is%20judge".
           STOP RUN.</pre></td></tr></table>

<p><a name="ada"></a><font size="2" color="#000080"><b>Ada</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">-- spawn curl, or use AWS.Client
-- AWS.Client.Get("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge")</pre></td></tr></table>

<p><a name="tcl"></a><font size="2" color="#000080"><b>Tcl</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">package require http
package require tls
::http::register https 443 ::tls::socket
set tok [::http::geturl https://marii-ask.mainline-mi.workers.dev/how%20is%20judge]
puts [::http::data $tok]</pre></td></tr></table>

<p><a name="awk"></a><font size="2" color="#000080"><b>Awk</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px"># gawk + curl
@load "readfile"
# or:
# curl -sS URL | awk '{ print }'</pre></td></tr></table>

<p><a name="vba"></a><font size="2" color="#000080"><b>VBA (Excel / Access)</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">Dim http As Object
Set http = CreateObject("MSXML2.XMLHTTP")
http.Open "GET", "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge", False
http.Send
MsgBox http.responseText</pre></td></tr></table>

<p><a name="ahk"></a><font size="2" color="#000080"><b>AutoHotkey</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">whr := ComObject("WinHttp.WinHttpRequest.5.1")
whr.Open("GET", "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge", true)
whr.Send()
whr.WaitForResponse()
MsgBox whr.ResponseText</pre></td></tr></table>

<p><a name="html"></a><font size="2" color="#000080"><b>HTML form</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#ffffcc" bordercolor="#cccc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">&lt;form action="https://marii-ask.mainline-mi.workers.dev/api/marii/ask" method="get"&gt;
  &lt;input name="q" value="how is judge"&gt;
  &lt;input type="submit" value="Ask"&gt;
&lt;/form&gt;</pre></td></tr></table>

<p><a name="cfml"></a><font size="2" color="#000080"><b>CFML / ColdFusion</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">&lt;cfhttp url="https://marii-ask.mainline-mi.workers.dev/how%20is%20judge" method="get" /&gt;
&lt;cfoutput&gt;#cfhttp.filecontent#&lt;/cfoutput&gt;</pre></td></tr></table>

<p><a name="plg"></a><font size="2" color="#000080"><b>Prolog</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">:- use_module(library(http/http_open)).
main :-
  http_open("https://marii-ask.mainline-mi.workers.dev/how%20is%20judge", In, []),
  copy_stream_data(In, user_output),
  close(In).</pre></td></tr></table>

<p><a name="lisp"></a><font size="2" color="#000080"><b>Common Lisp</b></font></p>
<table width="100%" border="1" cellpadding="6" cellspacing="0" bgcolor="#eeffee" bordercolor="#99cc99"><tr><td><pre style="margin:0;font-family:Courier New,Courier,monospace;font-size:12px">;; dexador
(print (dex:get "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"))</pre></td></tr></table>

<hr size="1" noshade color="#808080">
<p><font size="1" color="#333333">
GET only. POST is 501 on purpose (no cloud AI).<br>
Encode the question. Read the <b>reply</b> field from the JSON.<br>
These are copy-paste starters for <b>your</b> code, not a local VM lab.<br>
<a href="/mainlineintelligence#marii-ask">back to mainline intelligence</a>
&nbsp;|&nbsp;
<a href="https://marii-ask.mainline-mi.workers.dev/health">api health</a>
</font></p>

    </td>
  </tr>
</table>

<p><font face="Verdana, Arial, Helvetica, sans-serif" size="1" color="#333333">mainline intelligence -- marii ask -- pick a language, paste, go</font></p>

</font>
</body>
</html>
