/**
 * Pyx Assistant tests — run with:
 *   node public/betas/pyxassistant/pyx-assistant-slu.test.js
 */
"use strict";

var slu = require("./pyx-assistant-slu.js");
var i18n = require("./pyx-assistant-i18n.js");
var math = require("./pyx-assistant-math.js");
var kb = require("./pyx-assistant-kb.js");
var data = require("./kb/pyx-assistant-kb.json");

var failed = 0;

function assert(cond, msg) {
  if (!cond) {
    failed += 1;
    console.error("FAIL  " + msg);
  } else {
    console.log("ok    " + msg);
  }
}

var golden = slu.evaluateGolden();
assert(golden.accuracy === 1, "golden set accuracy is 100% (" + golden.hit + "/" + golden.total + ")");
if (golden.misses.length) {
  golden.misses.forEach(function (m) {
    console.error("      ", m.text, "expected", m.expected, "got", m.got);
  });
}

assert(slu.classify("hello").intent === "greet", "hello → greet");
assert(slu.classify("what's 9 * 8").intent === "calculator", "math → calculator");
assert(slu.classify("tell me a joke").intent === "joke", "joke intent");
assert(slu.classify("fun fact").intent === "fact", "fact intent");
assert(math.evalText("9*8") === 72, "9*8 = 72");
assert(math.evalText("fifteen percent of 80") === 12, "15% of 80 = 12");
assert(math.evalText("square root of 144") === 12, "sqrt 144");
assert(math.evalText("twenty plus twenty two") === 42, "word numbers");
assert(math.answer("32 f to c").value === 0, "32 F → 0 C");
assert(math.answer("5 km to miles").value > 3 && math.answer("5 km to miles").value < 3.2, "5 km → miles");
assert(math.looksMath("what's 12 times 4") === true, "looksMath times");
assert(slu.extractTheme("switch to mint") === "mint", "theme slot mint");
assert(slu.extractTheme("switch to calm contrast") === "calm-contrast", "theme slot calm contrast");
assert(slu.classify("switch to calm contrast").intent === "theme", "slu: calm contrast → theme");
assert(slu.extractLang("speak Spanish") === "es", "language slot es");

var resolved = slu.resolve(slu.classify("what time is it"), { lang: "en", t: i18n.t });
assert(typeof resolved.reply === "string" && resolved.reply.indexOf("It’s") === 0, "time reply is local");
assert(resolved.useLlm === false, "time does not call Talk");

var joke = slu.resolve(slu.classify("tell me a joke"), { lang: "en", t: i18n.t });
assert(joke.special === "__JOKE__", "jokes are local specials");
assert(joke.useLlm === false, "jokes do not call Talk");

var chat = slu.resolve(slu.classify("explain gravity"), { lang: "en", t: i18n.t });
assert(chat.useLlm === false, "open questions do not call Talk");

var weather = slu.resolve(slu.classify("weather in paris"), { lang: "en", t: i18n.t });
assert(weather.useWeb === true, "weather requests live data");
assert(!weather.reply, "weather leaves reply to the live path");
assert(slu.classify("what is MARII").intent === "marii", "slu: marii identity");
assert(slu.classify("what is Mainline Intelligence").intent === "mi", "slu: mi identity");
assert(/MARII|Mainline Artificial/i.test(slu.resolve(slu.classify("what is marii"), { lang: "en", t: i18n.t }).reply), "marii resolve");
assert(/beta|early/i.test(slu.resolve(slu.classify("is this a beta"), { lang: "en", t: i18n.t }).reply), "beta honesty");

var n = kb.load(data);
assert(n >= 1000, "knowledge base has 1000+ records (" + n + ")");
var grav = kb.retrieve("what is gravity", 0.4);
assert(grav && /gravity/.test(grav.rec.q) && grav.reply.length > 20, "retrieve gravity");
var photo = kb.retrieve("explain photosynthesis in simple terms", 0.35);
assert(photo && /photosynthesis/.test(photo.rec.q) && /light|plant/i.test(photo.reply), "retrieve photosynthesis");
assert(/honey|bees/i.test(kb.retrieve("fact about honey", 0.4).reply), "retrieve honey fact");
var j1 = kb.expandSpecial("__JOKE__");
var j2 = kb.expandSpecial("__JOKE__");
assert(j1 && j2 && j1.length > 10, "jokes resolve from the pack");
var haiku = kb.retrieve("write a haiku about rain", 0.62);
assert(!haiku || /rain|orb|pastel/i.test(haiku.reply), "haiku does not steal the email how-to");
assert(kb.warmFallback("write a haiku about rain").length > 20, "unmatched still gets a warm reply");
assert(i18n.t("en", "name") === "Pyx Assistant", "product name is Pyx Assistant");
assert(i18n.t("es", "name") === "Pyx Assistant", "name stays Pyx Assistant in ES");
assert(/local-first|MARII|cloud boost/i.test(i18n.t("en", "identity")), "identity is local-first MARII");

var learn = require("./pyx-assistant-learn.js");
var cookies = require("./pyx-assistant-cookies.js");

learn.reset();
var beforeJoke = learn.predict("tell me a joke").joke;
var i;
for (i = 0; i < 10; i++) learn.observe("tell me a joke", "joke");
assert(learn.predict("tell me a joke").joke > beforeJoke, "softmax learns joke preference");
assert(learn.kindFromIntent("calculator") === "math", "calculator maps to math class");
assert(learn.kindFromIntent("settings") === "talk", "settings maps to talk");

learn.reset();
var named = learn.ingest("my name is Brennan");
assert(learn.profile.name === "Brennan", "ingest name");
assert(/Brennan/.test(named.reply), "name confirmation");
learn.ingest("I like otters");
assert(learn.profile.likes[0] && /otter/i.test(learn.profile.likes[0]), "ingest likes");
learn.ingest("I hate riddles");
assert(learn.profile.dislikes.length >= 1, "ingest dislikes");
assert(/Brennan|otter/i.test(learn.summary()), "summary includes profile");
assert(/Brennan/.test(learn.greeting("Hi")), "greeting uses name");

var snap = learn.pack();
learn.reset();
assert(learn.profile.name === "", "reset clears name");
learn.unpack(snap);
assert(learn.profile.name === "Brennan", "unpack restores name");
assert(learn.profile.likes.length >= 1, "unpack restores likes");

var compact = cookies.compactMessages([
  { role: "user", content: "hi there" },
  { role: "assistant", content: "hello Brennan" },
  { role: "system", content: "skip me" },
]);
assert(compact.length === 2 && compact[0][0] === "u" && compact[1][0] === "a", "compact chats");
var expanded = cookies.expandMessages(compact);
assert(expanded[0].role === "user" && expanded[1].content === "hello Brennan", "expand chats");
assert(cookies.loadChats().length === 0, "loadChats is empty without document.cookie");
assert(cookies.loadModel() === null, "loadModel is null without document.cookie");

global.document = (function () {
  var store = {};
  return {
    get cookie() {
      return Object.keys(store)
        .filter(function (k) {
          return store[k];
        })
        .map(function (k) {
          return k + "=" + store[k];
        })
        .join("; ");
    },
    set cookie(s) {
      var nv = String(s).split(";")[0];
      var cut = nv.indexOf("=");
      var name = nv.slice(0, cut);
      var val = nv.slice(cut + 1);
      if (/max-age=0/.test(s)) store[name] = "";
      else store[name] = val;
    },
  };
})();
cookies.saveChats([
  { role: "user", content: "hello cookie" },
  { role: "assistant", content: "saved" },
]);
var fromCookie = cookies.loadChats();
assert(fromCookie.length === 2 && fromCookie[0].content === "hello cookie", "cookie roundtrip chats");
cookies.saveModel(learn.pack());
var ml = cookies.loadModel();
assert(ml && ml.v === 1 && ml.n === "Brennan", "cookie roundtrip model");
cookies.clearAll();
assert(cookies.loadChats().length === 0 && cookies.loadModel() === null, "clearAll wipes cookies");

kb.load(data);
var boosted = kb.retrieve("tell me something interesting", 0.2, { fact: 1 }, ["otters"]);
assert(boosted && boosted.rec, "retrieve still works with priors and likes");
assert(kb.family("joke_prompt") === "joke", "family maps joke_prompt");
assert(kb.family("define") === "fact", "family maps define to fact");
assert(i18n.t("en", "forgetMe").indexOf("Forget") !== -1, "forget-me string exists");
assert(slu.classify("show my data").intent === "data", "show my data → data");
assert(slu.resolve(slu.classify("show my data"), { lang: "en", t: i18n.t }).action.type === "data", "data intent opens Data");
assert(i18n.t("en", "dataTitle").indexOf("know") !== -1, "data title is plain language");

learn.reset();
var blank = learn.explain();
assert(blank.known === false && blank.patterned === false, "explain empty profile");
learn.ingest("my name is Brennan");
learn.ingest("I like otters");
var i2;
for (i2 = 0; i2 < 12; i2++) learn.observe("tell me a joke", "joke");
var report = learn.explain();
assert(report.name === "Brennan", "explain includes name");
assert(report.likes.length >= 1, "explain includes likes");
assert(report.patterned === true, "explain sees a taste pattern after jokes");
assert(report.tastes[0].id === "joke", "joke ranks first after joke observes");
assert(typeof report.tastes[0].amount === "string", "taste amount is a plain word");
assert(learn.kindFromIntent("data") === "talk", "data intent maps to talk class");

var sports = require("./pyx-assistant-sports.js");
assert(sports.looksSports("how's Ohtani doing") === true, "Ohtani looks like sports");
assert(sports.looksSports("tell me about gravity") === false, "gravity is not sports");
assert(sports.looksSports("hello") === false, "hello is not sports");
assert(sports.parse("Cubs score").kind === "scores", "Cubs score → scores");
assert(sports.parse("Cubs standings").kind === "standings", "Cubs standings");
assert(sports.looksSports("who's leading the nl central?") === true, "nl central looks like sports");
assert(sports.parse("who's leading the nl central?").kind === "standings", "nl central leading → standings");
assert(sports.parse("who's leading the nl central?").division && sports.parse("who's leading the nl central?").division.id === 205, "nl central division id");
assert(sports.parse("hr leader").kind === "leaders", "hr leader stays leaders");
assert(sports.parse("Judge vs Soto").kind === "compare", "Judge vs Soto → compare");
assert(sports.parse("nba scores").kind === "scores", "nba scores → scores");
assert(sports.parse("nba scores").league === "nba", "nba scores → nba league");
assert(sports.parse("nfl scores").league === "nfl", "nfl scores → nfl");
assert(sports.parse("how's LeBron doing").league === "nba", "LeBron → nba");
assert(sports.looksSports("nba scores") === true, "nba scores looks like sports");
assert(sports.looksSports("how's lebron doing") === true, "lebron looks like sports");
assert(sports.detectLeague("lakers score") === "nba", "lakers → nba");
assert(sports.detectLeague("Cubs score") === "mlb", "Cubs stay mlb");
var fakeLive = {
  status: { detailedState: "In Progress", abstractGameState: "Live" },
  teams: {
    away: { score: 1, team: { teamName: "Giants", name: "San Francisco Giants" } },
    home: { score: 0, team: { teamName: "Mets", name: "New York Mets" } },
  },
  linescore: {
    balls: 2,
    strikes: 1,
    outs: 1,
    inningState: "Bottom",
    currentInningOrdinal: "2nd",
    offense: { batter: { fullName: "Carson Benge" }, onDeck: { fullName: "Jared Young" } },
    defense: { pitcher: { fullName: "Anthony Molina" } },
  },
};
var liveLine = sports.formatGame(fakeLive);
assert(/2-1/.test(liveLine), "live line includes count");
assert(/Benge/.test(liveLine), "live line includes hitter");
assert(/Molina/.test(liveLine), "live line includes pitcher");
assert(/1 out/.test(liveLine), "live line includes outs");
var board = sports.fieldBoard(fakeLive);
assert(board.live === true, "field board is live");
assert(/Benge/.test(board.batter), "field board names the batter");
assert(/Molina/.test(board.pitcher), "field board names the pitcher");
assert(board.balls === 2 && board.strikes === 1 && board.outs === 1, "field board has the count");
assert(sports.parse("tennis scores").kind === "other", "tennis is unsupported");
assert(!sports.parse("who's pitching").team, "pitching is not Pirates");
assert(sports.parse("college football scores").kind === "scores", "cfb scores");
assert(sports.parse("college football scores").league === "ncaaf", "cfb → ncaaf");
assert(!sports.parse("college football scores").espnClub, "cfb is a league not a club");
assert(/star/i.test(sports.commentOPS("0.906")), "OPS .906 is star-level");
assert(slu.classify("how's Ohtani doing").intent === "sports", "slu: ohtani → sports");
assert(slu.classify("mlb scores").intent === "sports", "slu: mlb scores → sports");
assert(slu.classify("nba scores").intent === "sports", "slu: nba scores → sports");
assert(learn.kindFromIntent("sports") === "talk", "sports maps to talk class");
assert(/sports|MLB|NBA/i.test(i18n.t("en", "identity")), "identity mentions sports");

var wiki = require("./pyx-assistant-wiki.js");
assert(wiki.looksWikiWorthy("who is Ada Lovelace") === true, "wiki-worthy: Ada");
assert(wiki.looksWikiWorthy("tell me a joke") === false, "wiki skips jokes");
assert(wiki.extractTopic("who is Ada Lovelace") === "ada lovelace", "wiki topic extract");
assert(wiki.titleMatchScore("ada lovelace", "Ada Lovelace") >= 0.95, "wiki exact title");
assert(wiki.titleMatchScore("dogs", "Dog breeding") < 0.92, "wiki rejects loose title");
assert(wiki.firstSentences("One. Two. Three.", 2) === "One. Two.", "wiki two sentences");

if (failed) {
  console.error("\n" + failed + " failed");
  process.exit(1);
}
console.log("\nAll assistant tests passed.");
