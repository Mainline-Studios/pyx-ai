#!/usr/bin/env node
"use strict";

var path = require("path");
var failed = 0;

function assert(cond, msg) {
  if (!cond) {
    failed += 1;
    console.error("FAIL  " + msg);
  } else {
    console.log("ok    " + msg);
  }
}

var ask = require("../public/js/marii-ask.js");

(async function () {
  var echo = await ask.ask("hello from the try-it");
  assert(echo.ok === true, "stub echo ok");
  assert(echo.request === "hello from the try-it", "stub echo request");
  assert(echo.reply === "hello from the try-it", "stub echo reply without PA engines");
  assert(echo.backend === "local", "stub backend is local");

  var faq = await ask.ask("what is marii");
  assert(/Mainline Artificial Realtime Instant Intelligence/.test(faq.reply), "stub faq marii");

  var missing = await ask.ask("   ");
  assert(missing.ok === false, "blank is an error");

  var fromPath = ask.requestFromLocation({
    pathname: "/api/marii/ask/how%20is%20judge",
    search: "",
  });
  assert(fromPath === "how is judge", "path decode");

  var sports = require("../public/betas/pyxassistant/pyx-assistant-sports.js");
  var slu = require("../public/betas/pyxassistant/pyx-assistant-slu.js");
  var math = require("../public/betas/pyxassistant/pyx-assistant-math.js");
  var kb = require("../public/betas/pyxassistant/pyx-assistant-kb.js");
  var i18n = require("../public/betas/pyxassistant/pyx-assistant-i18n.js");

  assert(sports.looksSports("how is judge") === true, "how is judge looks like sports");
  var parsed = sports.parse("how is judge");
  assert(parsed.kind === "player", "how is judge parses as player");
  assert(parsed.names && parsed.names.join(" ").toLowerCase().indexOf("judge") !== -1, "parse keeps Judge");

  var realAnswer = sports.answer;
  sports.answer = async function () {
    return "Aaron Judge is hitting like a star this year — live card from the same sports path as Pyx Assistant.";
  };

  global.PyxAssistantSports = sports;
  global.PyxAssistantSLU = slu;
  global.PyxAssistantMath = math;
  global.PyxAssistantKB = kb;
  global.PyxAssistantI18n = i18n;

  var routed = await ask.ask("how is judge");
  assert(routed.ok === true, "PA ask ok");
  assert(routed.backend === "pyx-assistant", "uses pyx-assistant backend");
  assert(routed.source === "sports", "how is judge uses sports path");
  assert(/Aaron Judge/.test(routed.reply), "sports reply is not an echo");
  assert(routed.reply !== "how is judge", "does not echo how is judge");

  var mathHit = await ask.ask("what's 9 * 8");
  assert(mathHit.ok === true && /72/.test(mathHit.reply), "math uses PA calculator");

  var marii = await ask.ask("what is marii");
  assert(/Mainline Artificial Realtime Instant Intelligence/.test(marii.reply), "SLU marii identity");

  try {
    var pack = require(path.join(__dirname, "../public/betas/pyxassistant/kb/pyx-assistant-kb.json"));
    kb.load(pack);
    var grav = await ask.ask("what is gravity");
    assert(grav.ok === true && grav.reply && grav.reply !== "what is gravity", "KB gravity is a real answer");
  } catch (err) {
    console.log("skip  KB pack not built (" + err.message + ")");
  }

  sports.answer = realAnswer;

  if (failed) {
    console.error("\n" + failed + " failed");
    process.exit(1);
  }
  console.log("\nAll marii-ask js tests passed.");
})().catch(function (err) {
  console.error(err);
  process.exit(1);
});
