/**
 * Pyx Assistant — live sports. MLB Stats API for baseball boards and
 * player cards; ESPN public scoreboard for NBA, NFL, NHL, WNBA, CFB, MLS, EPL.
 */
(function (root) {
  "use strict";

  var MLB = "https://statsapi.mlb.com/api/v1";
  var DIVISIONS = {
    201: "AL East",
    202: "AL Central",
    200: "AL West",
    204: "NL East",
    205: "NL Central",
    203: "NL West",
  };
  var NICK = {
    ohtani: "Shohei Ohtani",
    shohei: "Shohei Ohtani",
    judge: "Aaron Judge",
    soto: "Juan Soto",
    harper: "Bryce Harper",
    trout: "Mike Trout",
    acuna: "Ronald Acuna Jr.",
    "acuna jr": "Ronald Acuna Jr.",
    betts: "Mookie Betts",
    mookie: "Mookie Betts",
    tatis: "Fernando Tatis Jr.",
    witt: "Bobby Witt Jr.",
    elly: "Elly De La Cruz",
    "de la cruz": "Elly De La Cruz",
    schwarber: "Kyle Schwarber",
    yordan: "Yordan Alvarez",
    skubal: "Tarik Skubal",
    cole: "Gerrit Cole",
    tucker: "Kyle Tucker",
    riley: "Austin Riley",
    pca: "Pete Crow-Armstrong",
    "crow-armstrong": "Pete Crow-Armstrong",
    "crow armstrong": "Pete Crow-Armstrong",
    gunnar: "Gunnar Henderson",
    ragans: "Cole Ragans",
    skenes: "Paul Skenes",
    yamamoto: "Yoshinobu Yamamoto",
    lebron: "LeBron James",
    jokic: "Nikola Jokic",
    luka: "Luka Doncic",
    curry: "Stephen Curry",
    giannis: "Giannis Antetokounmpo",
    tatum: "Jayson Tatum",
    mahomes: "Patrick Mahomes",
    mcdavid: "Connor McDavid",
    haaland: "Erling Haaland",
    wemby: "Victor Wembanyama",
  };
  var TEAM_ALIASES = [
    ["new york yankees", "yankees", "nyy"],
    ["boston red sox", "red sox", "bos"],
    ["tampa bay rays", "rays", "tb"],
    ["toronto blue jays", "blue jays", "jays", "tor"],
    ["baltimore orioles", "orioles", "bal"],
    ["chicago white sox", "white sox", "cws"],
    ["cleveland guardians", "guardians", "cle"],
    ["detroit tigers", "tigers", "det"],
    ["kansas city royals", "royals", "kc"],
    ["minnesota twins", "twins", "min"],
    ["houston astros", "astros", "hou"],
    ["seattle mariners", "mariners", "sea"],
    ["texas rangers", "rangers", "tex"],
    ["athletics", "a's", "oakland", "oakland athletics", "ath"],
    ["los angeles angels", "anaheim angels", "angels", "laa"],
    ["philadelphia phillies", "phillies", "phi"],
    ["new york mets", "mets", "nym"],
    ["atlanta braves", "braves", "atl"],
    ["miami marlins", "marlins", "mia"],
    ["washington nationals", "nationals", "nats", "was"],
    ["milwaukee brewers", "brewers", "mil"],
    ["chicago cubs", "cubs", "chc"],
    ["cincinnati reds", "reds", "cin"],
    ["pittsburgh pirates", "pirates", "pit"],
    ["st louis cardinals", "st. louis cardinals", "cardinals", "cards", "stl"],
    ["los angeles dodgers", "dodgers", "lad"],
    ["san diego padres", "padres", "sd"],
    ["arizona diamondbacks", "diamondbacks", "dbacks", "d-backs", "ari"],
    ["san francisco giants", "giants", "sf"],
    ["colorado rockies", "rockies", "col"],
  ];
  var STOP = {
    how: 1, is: 1, are: 1, was: 1, were: 1, the: 1, a: 1, an: 1, of: 1, for: 1,
    about: 1, tell: 1, me: 1, please: 1, what: 1, whats: 1, who: 1, whom: 1,
    his: 1, her: 1, their: 1, he: 1, she: 1, they: 1, him: 1, them: 1, that: 1,
    this: 1, last: 1, year: 1, years: 1, season: 1, career: 1, lifetime: 1,
    stats: 1, stat: 1, numbers: 1, doing: 1, did: 1, does: 1, look: 1, up: 1,
    vs: 1, versus: 1, compared: 1, compare: 1, to: 1, with: 1, and: 1, or: 1,
    mlb: 1, baseball: 1, player: 1, players: 1, team: 1, game: 1, games: 1,
    today: 1, tonight: 1, live: 1, current: 1, now: 1, just: 1, like: 1,
    hitting: 1, pitching: 1, batting: 1, average: 1, ops: 1, era: 1, rbi: 1,
    rbis: 1, homers: 1, homer: 1, home: 1, runs: 1, stolen: 1, bases: 1,
    whip: 1, strikeouts: 1, wins: 1, innings: 1, good: 1, bad: 1, still: 1,
    score: 1, scores: 1, standings: 1, record: 1, got: 1, get: 1, give: 1,
    show: 1, pull: 1, check: 1, on: 1, in: 1, at: 1, from: 1, by: 1,
    much: 1, many: 1, has: 1, have: 1, been: 1, going: 1, so: 1, far: 1,
    junior: 1, jr: 1, pyx: 1, can: 1, you: 1, your: 1, my: 1,
    nba: 1, nfl: 1, nhl: 1, wnba: 1, mls: 1, soccer: 1, football: 1,
    basketball: 1, hockey: 1, baseball: 1,
  };
  var SPORT_RE =
    /\b(mlb|baseball|world series|pennant|ops|obp|slg|era\b|whip|rbi|rbis|batting average|on.?base|slugging|home runs?|homers?|stolen bases?|strikeouts|innings pitched|box score|scoreboard|standings|cy young|mvp|two-?way|pitcher|hitter|batter|inning|bullpen|lineup|dodgers|yankees|cubs|red sox|mets|braves|phillies|giants|padres|astros|rangers|mariners|angels|orioles|rays|jays|guardians|tigers|twins|royals|white sox|athletics|nationals|marlins|brewers|reds|pirates|cardinals|rockies|diamondbacks|d-backs|nba|nfl|nhl|wnba|mls|soccer|premier league|basketball|hockey|football|lakers|celtics|warriors|knicks|chiefs|cowboys|patriots|maple leafs|canadiens)\b/i;
  var FOLLOW_RE =
    /\b(he|him|his|she|her|they|them|their|that guy|same guy|career|last year|this year|this season|ops|era|whip|average|homers?|rbi|how('s| is) (he|she|that)|is that good|was that good|and last|what about his|how about his|who('s| is) (up|pitching|batting|hitting|in)|the count|how many outs|runners|on base|what('s| is) the (count|score)|who has the ball|what down|possession)\b/i;
  var OTHER_RE = /\b(nba|nfl|nhl|mls|soccer|premier league|epl|basketball|football|hockey|wnba|ncaaf|college football)\b/i;
  var UNSUPPORTED_RE = /\b(f1|formula 1|tennis|golf|pga|cricket|ufc|mma|boxing)\b/i;
  var ESPN = "https://site.web.api.espn.com/apis";
  var ESPN_BOARDS = {
    nba: { path: "/site/v2/sports/basketball/nba/scoreboard", sport: "basketball", slug: "nba", label: "NBA" },
    wnba: { path: "/site/v2/sports/basketball/wnba/scoreboard", sport: "basketball", slug: "wnba", label: "WNBA" },
    nfl: { path: "/site/v2/sports/football/nfl/scoreboard", sport: "football", slug: "nfl", label: "NFL" },
    ncaaf: { path: "/site/v2/sports/football/college-football/scoreboard", sport: "football", slug: "college-football", label: "college football" },
    nhl: { path: "/site/v2/sports/hockey/nhl/scoreboard", sport: "hockey", slug: "nhl", label: "NHL" },
    mls: { path: "/site/v2/sports/soccer/usa.1/scoreboard", sport: "soccer", slug: "usa.1", label: "MLS" },
    epl: { path: "/site/v2/sports/soccer/eng.1/scoreboard", sport: "soccer", slug: "eng.1", label: "Premier League" },
  };
  var ESPN_CLUBS = [
    { league: "nba", names: ["trail blazers", "timberwolves", "76ers", "sixers", "lakers", "celtics", "warriors", "knicks", "miami heat", "heat", "bucks", "nuggets", "suns", "raptors", "thunder", "spurs", "mavericks", "mavs", "rockets", "clippers", "bulls", "brooklyn nets", "hawks", "pistons", "pacers", "orlando magic", "hornets", "wizards", "kings", "utah jazz", "pelicans", "grizzlies"] },
    { league: "wnba", names: ["las vegas aces", "aces", "liberty", "fever", "lynx", "seattle storm"] },
    { league: "nfl", names: ["new york giants", "ny giants", "san francisco 49ers", "49ers", "niners", "chiefs", "cowboys", "eagles", "packers", "patriots", "seahawks", "ravens", "bills", "lions", "bengals", "dolphins", "vikings", "bears", "jets", "steelers", "broncos", "chargers", "raiders", "rams", "saints", "falcons", "texans", "colts", "jaguars", "titans", "commanders", "browns", "carolina panthers"] },
    { league: "nhl", names: ["maple leafs", "leafs", "canadiens", "habs", "bruins", "oilers", "penguins", "blackhawks", "red wings", "new york rangers", "ny rangers", "golden knights", "avalanche", "lightning", "capitals", "hurricanes"] },
    { league: "mls", names: ["inter miami", "la galaxy", "lafc", "atlanta united", "seattle sounders"] },
    { league: "epl", names: ["manchester city", "man city", "manchester united", "man united", "arsenal", "liverpool", "chelsea", "tottenham", "newcastle"] },
    { league: "ncaaf", names: ["crimson tide", "georgia bulldogs", "ohio state"] },
  ];

  var cache = { roster: null, teamById: {}, rosterAt: 0, player: {}, espn: {} };
  var ctx = {
    player: null,
    team: null,
    lastStat: null,
    lastAsk: "",
    lastLeague: "mlb",
    lastGamePk: null,
    espnPlayer: null,
    espnClub: null,
    board: null,
  };

  function seasonYear() {
    var d = new Date();
    return d.getMonth() >= 2 ? d.getFullYear() : d.getFullYear() - 1;
  }

  function norm(s) {
    return String(s || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/['’.]/g, "")
      .replace(/[^a-z0-9\s-]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function mlb(path) {
    var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    var t = setTimeout(function () {
      if (ctrl) ctrl.abort();
    }, 9000);
    return fetch(MLB + path, { signal: ctrl ? ctrl.signal : undefined })
      .then(function (res) {
        if (!res.ok) throw new Error("mlb " + res.status);
        return res.json();
      })
      .finally(function () {
        clearTimeout(t);
      });
  }

  function espn(path) {
    var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    var t = setTimeout(function () {
      if (ctrl) ctrl.abort();
    }, 9000);
    return fetch(ESPN + path, { signal: ctrl ? ctrl.signal : undefined })
      .then(function (res) {
        if (!res.ok) throw new Error("espn " + res.status);
        return res.json();
      })
      .finally(function () {
        clearTimeout(t);
      });
  }

  function todayISO() {
    var d = new Date();
    var m = String(d.getMonth() + 1);
    var day = String(d.getDate());
    if (m.length < 2) m = "0" + m;
    if (day.length < 2) day = "0" + day;
    return d.getFullYear() + "-" + m + "-" + day;
  }

  function hasClubName(n, name) {
    if (name.indexOf(" ") !== -1) return n.indexOf(name) !== -1;
    return new RegExp("\\b" + name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b").test(n);
  }

  function findEspnClub(low) {
    var n = norm(low);
    var best = null;
    var bestLen = 0;
    ESPN_CLUBS.forEach(function (row) {
      row.names.forEach(function (name) {
        if (hasClubName(n, name) && name.length > bestLen) {
          bestLen = name.length;
          best = { league: row.league, needle: name };
        }
      });
    });
    return best;
  }

  function detectLeague(low) {
    var n = String(low || "").toLowerCase();
    if (/\b(premier league|\bepl\b)\b/.test(n)) return "epl";
    if (/\b(mls|soccer)\b/.test(n) && !/\b(mlb|baseball)\b/.test(n)) return "mls";
    if (/\bwnba\b/.test(n)) return "wnba";
    if (/\b(ncaaf|college football)\b/.test(n)) return "ncaaf";
    if (/\b(nhl|hockey)\b/.test(n)) return "nhl";
    if (/\b(nfl|touchdown|quarterback)\b/.test(n)) return "nfl";
    if (/\bfootball\b/.test(n) && !/\b(soccer|college)\b/.test(n)) return "nfl";
    if (/\b(nba|basketball)\b/.test(n)) return "nba";
    if (/\b(lebron|jokic|luka|giannis|curry|tatum|wemby|wembanyama)\b/.test(n)) return "nba";
    if (/\bmahomes\b/.test(n)) return "nfl";
    if (/\bmcdavid\b/.test(n)) return "nhl";
    if (/\bhaaland\b/.test(n)) return "epl";
    var club = findEspnClub(n);
    if (club) return club.league;
    if (/\b(mlb|baseball)\b/.test(n) || findTeam(n)) return "mlb";
    if (ctx.lastLeague) return ctx.lastLeague;
    return "mlb";
  }

  function aliasHits(n, alias) {
    var a = norm(alias);
    if (!a) return false;
    if (a.length <= 3) {
      return new RegExp("\\b" + a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b").test(n);
    }
    return n.indexOf(a) !== -1;
  }

  function findTeam(low) {
    var n = norm(low);
    if (/\bwhite sox\b/.test(n) || /\bcws\b/.test(n)) return "white sox";
    if (/\bred sox\b/.test(n) || /\bbos\b/.test(n)) return "red sox";
    var i, aliases, a;
    for (i = 0; i < TEAM_ALIASES.length; i++) {
      aliases = TEAM_ALIASES[i];
      for (a = 0; a < aliases.length; a++) {
        if (aliasHits(n, aliases[a])) return aliases[0];
      }
    }
    return "";
  }

  function extractYear(low) {
    var m = String(low).match(/\b(20\d{2})\b/);
    return m ? m[1] : "";
  }

  function extractWindow(low) {
    if (/\b(career|lifetime|all[- ]time)\b/.test(low)) return "career";
    if (/\blast year\b/.test(low)) return "last";
    if (extractYear(low)) return "year";
    return "season";
  }

  function extractFocus(low) {
    if (/\b(era|whip|innings|starter|starts|saves|ks\b|strikeouts|on the mound|pitching)\b/.test(low)) {
      return "pitching";
    }
    if (/\b(ops|obp|slg|batting|hitting|homers?|home runs?|rbi|stolen|average at the plate)\b/.test(low)) {
      return "hitting";
    }
    return "";
  }

  function extractStat(low) {
    if (/\bops\b/.test(low)) return "ops";
    if (/\b(obp|on.?base)\b/.test(low)) return "obp";
    if (/\b(slg|slugging)\b/.test(low)) return "slg";
    if (/\b(batting average|ba\b|avg\b|hitting\.\d)\b/.test(low) || /\baverage\b/.test(low)) return "avg";
    if (/\b(home runs?|homers?|hrs?\b)\b/.test(low)) return "homeRuns";
    if (/\brbis?\b/.test(low)) return "rbi";
    if (/\bstolen bases?\b/.test(low)) return "stolenBases";
    if (/\bera\b/.test(low)) return "era";
    if (/\bwhip\b/.test(low)) return "whip";
    if (/\b(strikeouts|k's|ks\b)\b/.test(low)) return "strikeOuts";
    if (/\bwins?\b/.test(low)) return "wins";
    if (/\binnings\b/.test(low)) return "inningsPitched";
    return "";
  }

  function extractNames(raw) {
    var low = norm(raw);
    var found = [];
    var nick;
    var keys = Object.keys(NICK).sort(function (a, b) {
      return b.length - a.length;
    });
    keys.forEach(function (k) {
      if (low.indexOf(k) !== -1 && found.indexOf(NICK[k]) === -1) found.push(NICK[k]);
    });
    var vs = raw.match(
      /([A-Za-z][A-Za-z.'-]+(?:\s+[A-Za-z][A-Za-z.'-]+){0,3})\s+(?:vs\.?|versus|compared to|compare(?:d)? to)\s+([A-Za-z][A-Za-z.'-]+(?:\s+[A-Za-z][A-Za-z.'-]+){0,3})/i
    );
    if (vs) {
      [vs[1], vs[2]].forEach(function (part) {
        var cleaned = cleanName(part);
        if (cleaned && found.indexOf(cleaned) === -1) found.push(cleaned);
      });
    }
    if (found.length) return found.slice(0, 2);
    var leftover = cleanName(raw);
    return leftover ? [leftover] : [];
  }

  function cleanName(text) {
    var parts = norm(text)
      .split(" ")
      .filter(function (w) {
        return w && !STOP[w] && !/^\d+$/.test(w) && w.length > 1;
      });
    TEAM_ALIASES.forEach(function (aliases) {
      aliases.forEach(function (a) {
        var bits = norm(a).split(" ");
        if (bits.length === 1) {
          parts = parts.filter(function (p) {
            return p !== bits[0];
          });
        }
      });
    });
    if (!parts.length || parts.length > 4) return "";
    return parts
      .map(function (w) {
        if (w === "jr") return "Jr.";
        return w.charAt(0).toUpperCase() + w.slice(1);
      })
      .join(" ");
  }

  function isTeamQuery(names, team, low) {
    if (!team) return false;
    if (/\b(stats?|ops|era|hitting|pitching|batting|player)\b/.test(low)) return false;
    if (!names.length) return true;
    if (names.length > 1) return false;
    var last = norm(team).split(" ").pop();
    return norm(names[0]) === last || norm(team).indexOf(norm(names[0])) !== -1;
  }

  function looksSports(text) {
    var raw = String(text || "");
    var low = raw.toLowerCase();
    if (SPORT_RE.test(low) || findTeam(low) || OTHER_RE.test(low) || UNSUPPORTED_RE.test(low) || findEspnClub(low)) return true;
    if ((ctx.player || ctx.espnPlayer || ctx.lastGamePk || ctx.team) && FOLLOW_RE.test(low)) return true;
    var n;
    for (n in NICK) {
      if (Object.prototype.hasOwnProperty.call(NICK, n) && low.indexOf(n) !== -1) return true;
    }
    if (/\b(stats?|how('s| is| did)|tell me about|look up|compare)\b/i.test(low)) {
      var maybe = extractNames(raw);
      if (maybe.length && (maybe[0].split(" ").length >= 2 || NICK[norm(maybe[0].toLowerCase())])) return true;
    }
    return false;
  }

  function liveAsk(low) {
    return /\b(who('s| is) (up|pitching|batting|hitting)|the count|how many outs|runners|on base|what('s| is) the (count|score)|who has the ball|what down|possession)\b/.test(low);
  }

  function parse(text) {
    var raw = String(text || "").trim();
    var low = raw.toLowerCase();
    var names = extractNames(raw);
    var team = findTeam(low);
    var espnClub = findEspnClub(low);
    var league = detectLeague(low);
    var other = OTHER_RE.test(low);
    var kind = "player";
    if (UNSUPPORTED_RE.test(low)) {
      kind = "other";
    } else {
      if (/\b(standings|in first|playoff|wild card|division race|what place)\b/.test(low)) kind = "standings";
      if (/\b(score|scores|who won|who('s| is) winning|scoreboard|game tonight|did they win)\b/.test(low) || liveAsk(low)) {
        kind = "scores";
      }
      if (/\b(lead(s|ing) the|leaderboard|most home runs|hr leader|batting title|cy young)\b/.test(low)) {
        kind = "leaders";
      }
      if (names.length >= 2 || /\b(vs\.?|versus|compared to|compare)\b/.test(low)) kind = "compare";
      if (kind === "player" && team && isTeamQuery(names, team, low)) {
        kind = /\b(standings|place|record|how are)\b/.test(low) ? "standings" : "scores";
      }
      if (kind === "player" && espnClub && isTeamQuery(names, espnClub.needle, low)) kind = "scores";
      if (kind === "player" && !names.length && espnClub) kind = "scores";
      if (kind === "player" && !names.length && ctx.player && FOLLOW_RE.test(low)) kind = "follow";
      if (kind === "player" && !names.length && ctx.espnPlayer && FOLLOW_RE.test(low)) kind = "follow";
    }
    if (kind === "scores" && !team && ctx.team && league === "mlb" && ctx.lastLeague === "mlb" && !/\b(all |every |mlb scores|scoreboard)\b/.test(low)) {
      team = ctx.team;
    }
    if (kind === "scores" && !espnClub && ctx.espnClub && league !== "mlb" && ctx.lastLeague === league) {
      espnClub = ctx.espnClub;
    }
    return {
      raw: raw,
      low: low,
      names: names,
      team: team,
      espnClub: espnClub,
      league: league,
      year: extractYear(low),
      window: extractWindow(low),
      focus: extractFocus(low),
      stat: extractStat(low),
      kind: kind,
      other: other,
      goodQ: /\bis that (any )?good\b/.test(low) || /\bwas that good\b/.test(low),
    };
  }

  async function ensureIndex() {
    if (cache.roster && Date.now() - cache.rosterAt < 30 * 60 * 1000) return;
    var year = seasonYear();
    var pack = await Promise.all([mlb("/sports/1/players?season=" + year), mlb("/teams?sportId=1&season=" + year)]);
    cache.roster = pack[0].people || [];
    cache.teamById = {};
    (pack[1].teams || []).forEach(function (t) {
      cache.teamById[t.id] = t;
    });
    cache.rosterAt = Date.now();
  }

  function scoreName(person, query) {
    var q = norm(query);
    var full = norm(person.fullName);
    var last = norm(person.lastName);
    var first = norm(person.firstName);
    if (!q) return 0;
    if (full === q) return 120;
    if (last === q && first) return 88;
    if ((first + " " + last) === q) return 118;
    if (full.indexOf(q) === 0) return 80;
    if (last.indexOf(q) === 0 && q.length >= 4) return 70;
    if (full.indexOf(q) !== -1) return 50;
    return 0;
  }

  async function findPlayer(query) {
    var q = NICK[norm(query)] || query;
    await ensureIndex();
    var best = null;
    var bestScore = 0;
    (cache.roster || []).forEach(function (p) {
      var s = scoreName(p, q);
      if (s > bestScore) {
        bestScore = s;
        best = p;
      }
    });
    if (best && bestScore >= 70) return best;
    var data = await mlb("/people/search?names=" + encodeURIComponent(q));
    var people = data.people || [];
    people.forEach(function (p) {
      var s = scoreName(p, q) + (p.primaryPosition && p.primaryPosition.abbreviation !== "X" ? 5 : 0);
      if (s > bestScore) {
        bestScore = s;
        best = p;
      }
    });
    return bestScore >= 50 ? best : null;
  }

  function teamName(person) {
    var id = person && person.currentTeam && person.currentTeam.id;
    var t = id && cache.teamById[id];
    return (t && (t.teamName || t.name)) || "";
  }

  async function loadCard(person) {
    var id = person.id;
    var hit = cache.player[id];
    if (hit && Date.now() - hit.at < 8 * 60 * 1000) return hit.data;
    var year = seasonYear();
    var data = await mlb(
      "/people/" +
        id +
        "?hydrate=currentTeam,stats(group=[hitting,pitching],type=[season,career,yearByYear])"
    );
    var p = (data.people || [])[0];
    if (!p) return null;
    var card = {
      id: p.id,
      name: p.fullName,
      pos: (p.primaryPosition && p.primaryPosition.abbreviation) || "",
      posName: (p.primaryPosition && p.primaryPosition.name) || "",
      number: p.primaryNumber || "",
      team: (p.currentTeam && p.currentTeam.name) || teamName(p),
      bats: (p.batSide && p.batSide.description) || "",
      throws: (p.pitchHand && p.pitchHand.description) || "",
      hitting: { season: null, career: null, years: {} },
      pitching: { season: null, career: null, years: {} },
      season: year,
    };
    (p.stats || []).forEach(function (block) {
      var group = ((block.group && block.group.displayName) || "").toLowerCase();
      var type = ((block.type && block.type.displayName) || "").toLowerCase();
      var bucket = group.indexOf("pitch") === 0 ? card.pitching : group.indexOf("hit") === 0 ? card.hitting : null;
      if (!bucket) return;
      (block.splits || []).forEach(function (sp) {
        if (type === "season") bucket.season = sp.stat;
        else if (type === "career") bucket.career = sp.stat;
        else if (type === "yearbyyear" && sp.season) bucket.years[sp.season] = sp.stat;
      });
    });
    cache.player[id] = { at: Date.now(), data: card };
    return card;
  }

  function splitFor(bucket, window, year) {
    if (!bucket) return null;
    if (window === "career") return { label: "for his career", stat: bucket.career };
    if (window === "last") {
      var y = String(seasonYear() - 1);
      return { label: "in " + y, stat: bucket.years[y] || null };
    }
    if (window === "year" && year) {
      return { label: "in " + year, stat: bucket.years[year] || (year === String(seasonYear()) ? bucket.season : null) };
    }
    return { label: "this season", stat: bucket.season };
  }

  function commentOPS(ops) {
    var n = parseFloat(ops);
    if (isNaN(n)) return "";
    if (n >= 1) return "That’s MVP-ish production.";
    if (n >= 0.9) return "That’s star-level.";
    if (n >= 0.8) return "That’s a genuinely good bat.";
    if (n >= 0.72) return "Around a solid big-league bat.";
    if (n >= 0.65) return "A bit light for a regular.";
    return "That’s been a struggle at the plate.";
  }

  function commentAVG(avg) {
    var n = parseFloat(avg);
    if (isNaN(n)) return "";
    if (n >= 0.3) return "That’s a batting-title pace.";
    if (n >= 0.27) return "That’s a real hit tool.";
    if (n >= 0.24) return "Playable, not loud.";
    return "The average has been skinny.";
  }

  function commentERA(era) {
    var n = parseFloat(era);
    if (isNaN(n)) return "";
    if (n <= 2.5) return "That’s ace-level run prevention.";
    if (n <= 3.3) return "That’s a strong starter’s ERA.";
    if (n <= 4.2) return "That’s roughly in the mix.";
    return "That’s been a rough year on the mound.";
  }

  function commentStat(key, value) {
    if (key === "ops") return commentOPS(value);
    if (key === "avg") return commentAVG(value);
    if (key === "era") return commentERA(value);
    if (key === "homeRuns") {
      var hr = parseInt(value, 10);
      if (hr >= 40) return "That’s a 40-homer season. Loud.";
      if (hr >= 25) return "That’s real thump.";
      if (hr >= 15) return "Decent pop.";
      return "Not a huge homer year.";
    }
    return "";
  }

  function hittingBits(st) {
    if (!st) return "";
    return (
      "hitting " +
      st.avg +
      " with " +
      st.homeRuns +
      " homers, " +
      st.rbi +
      " RBIs, and a " +
      st.ops +
      " OPS in " +
      st.gamesPlayed +
      " games"
    );
  }

  function pitchingBits(st) {
    if (!st) return "";
    return (
      st.era +
      " ERA, " +
      st.whip +
      " WHIP, " +
      (st.wins != null ? st.wins + "-" + st.losses : "") +
      " in " +
      st.inningsPitched +
      " IP with " +
      st.strikeOuts +
      " Ks"
    );
  }

  function isPitcher(card) {
    return card.pos === "P" || card.pos === "SP" || card.pos === "RP";
  }

  function isTwoWay(card) {
    return card.pos === "TWP" || (card.hitting.season && card.pitching.season && card.pitching.season.gamesStarted);
  }

  function remember(card, key, value) {
    ctx.player = card;
    ctx.lastStat = key && value != null ? { key: key, value: value } : ctx.lastStat;
  }

  function playerReply(card, q) {
    var wantPitch = q.focus === "pitching" || (isPitcher(card) && q.focus !== "hitting");
    var wantHit = q.focus === "hitting" || (!isPitcher(card) && q.focus !== "pitching") || isTwoWay(card);
    if (q.stat === "era" || q.stat === "whip" || q.stat === "wins" || q.stat === "inningsPitched") wantPitch = true;
    if (q.stat === "ops" || q.stat === "avg" || q.stat === "homeRuns" || q.stat === "rbi" || q.stat === "obp" || q.stat === "slg") {
      wantHit = true;
      wantPitch = q.focus === "pitching";
    }
    var hit = splitFor(card.hitting, q.window, q.year);
    var pit = splitFor(card.pitching, q.window, q.year);
    var team = card.team ? " for the " + card.team : "";
    var who = card.name + (card.number ? " (#" + card.number + ")" : "") + team;
    var bits = [];
    var st;

    if (q.goodQ && ctx.lastStat) {
      var note = commentStat(ctx.lastStat.key, ctx.lastStat.value);
      return note
        ? "Yeah — " + ctx.lastStat.value + " " + ctx.lastStat.key.replace("homeRuns", "homers") + ". " + note + " Want last year or career next?"
        : "It’s a useful number, but I’d stack it next to last year to be sure. Want me to?";
    }

    if (q.stat && wantHit && hit && hit.stat && hit.stat[q.stat] != null) {
      st = hit.stat;
      remember(card, q.stat, st[q.stat]);
      bits.push(card.name + " is at " + st[q.stat] + " " + hit.label + " (" + q.stat.replace("homeRuns", "home runs") + ").");
      bits.push(commentStat(q.stat, st[q.stat]));
      if (q.stat !== "ops" && st.ops) bits.push("OPS sits at " + st.ops + ".");
      bits.push("I can do career, last year, or stack him against someone.");
      return bits.filter(Boolean).join(" ") + " =)";
    }
    if (q.stat && wantPitch && pit && pit.stat && pit.stat[q.stat] != null) {
      st = pit.stat;
      remember(card, q.stat, st[q.stat]);
      bits.push(card.name + " is at " + st[q.stat] + " " + pit.label + " on the mound.");
      bits.push(commentStat(q.stat, st[q.stat]));
      bits.push("Want the full pitching line, or a hitter comparison?");
      return bits.filter(Boolean).join(" ") + " =)";
    }

    if (isTwoWay(card) && q.focus !== "hitting" && q.focus !== "pitching") {
      var h = hit && hit.stat ? hittingBits(hit.stat) : "";
      var p = pit && pit.stat ? pitchingBits(pit.stat) : "";
      remember(card, h && hit.stat ? "ops" : "era", h && hit.stat ? hit.stat.ops : pit && pit.stat ? pit.stat.era : null);
      return (
        who +
        " is actually two-way " +
        (hit ? hit.label : "this year") +
        ". " +
        (h ? "At the plate he’s " + h + ". " : "") +
        (p ? "On the mound: " + p + ". " : "") +
        (h && hit.stat ? commentOPS(hit.stat.ops) + " " : "") +
        "Ask for just hitting, just pitching, career, or a compare. =)"
      );
    }

    if (wantPitch && pit && pit.stat) {
      remember(card, "era", pit.stat.era);
      return (
        who +
        " on the mound " +
        pit.label +
        ": " +
        pitchingBits(pit.stat) +
        ". " +
        commentERA(pit.stat.era) +
        " I can pull career, last year, or how the Ks look. =)"
      );
    }
    if (hit && hit.stat) {
      remember(card, "ops", hit.stat.ops);
      return (
        who +
        " is " +
        hittingBits(hit.stat) +
        " " +
        hit.label +
        ". " +
        commentOPS(hit.stat.ops) +
        " Want career, last year, or a compare? =)"
      );
    }
    return (
      "I found " +
      card.name +
      team +
      ", but that slice of the stat sheet is empty — maybe they didn’t play that year. Try this season or career. =)"
    );
  }

  async function talkPlayer(q) {
    if (q.kind === "follow" && ctx.espnPlayer && ctx.lastLeague !== "mlb") {
      return formatEspnAthlete(ctx.espnPlayer, q);
    }
    var name = q.kind === "follow" && ctx.player ? ctx.player.name : q.names[0];
    if (!name) {
      if (q.league && q.league !== "mlb") return espnScores(q);
      return "Name a player and I’ll pull live numbers — Ohtani, LeBron, Mahomes, a Cubs starter, whoever. Baseball is still the deepest feed. =)";
    }
    if (q.league && q.league !== "mlb") {
      var other = await espnPlayer(q);
      if (other) return other;
      return (
        "I couldn’t find “" +
        name +
        "” on the " +
        ((ESPN_BOARDS[q.league] && ESPN_BOARDS[q.league].label) || q.league) +
        " list. Try a full name. =)"
      );
    }
    var person = await findPlayer(name);
    if (!person) {
      var espnTry = await espnPlayer(q);
      if (espnTry) return espnTry;
      return (
        "I couldn’t find “" +
        name +
        "” on the live lists I use. Try a full name, or a club — Dodgers, Lakers, Chiefs. =)"
      );
    }
    var card = await loadCard(person);
    if (!card) return "I found the name, then the stat pack blanked on me. Try once more. =)";
    ctx.player = card;
    ctx.lastLeague = "mlb";
    ctx.espnPlayer = null;
    ctx.lastAsk = q.raw;
    return playerReply(card, q);
  }

  async function compare(q) {
    var names = q.names.slice(0, 2);
    if (names.length < 2 && ctx.player && names.length === 1) names = [ctx.player.name, names[0]];
    if (names.length < 2) return "Give me two names — like Judge vs Soto — and I’ll stack the live MLB lines. =)";
    var people = await Promise.all(names.map(findPlayer));
    if (!people[0] || !people[1]) {
      return "I need two MLB names I can actually resolve. Try “Judge vs Soto”. =)";
    }
    var cards = await Promise.all(people.map(loadCard));
    var a = cards[0];
    var b = cards[1];
    var usePitch = q.focus === "pitching" || (isPitcher(a) && isPitcher(b));
    var sa = splitFor(usePitch ? a.pitching : a.hitting, q.window, q.year);
    var sb = splitFor(usePitch ? b.pitching : b.hitting, q.window, q.year);
    if (!sa.stat || !sb.stat) return "One of those lines is empty for that season. Try this year. =)";
    ctx.player = a;
    if (usePitch) {
      return (
        a.name +
        " vs " +
        b.name +
        " " +
        sa.label +
        " on the mound: " +
        a.name +
        " has a " +
        sa.stat.era +
        " ERA and " +
        sa.stat.strikeOuts +
        " Ks; " +
        b.name +
        " is at " +
        sb.stat.era +
        " with " +
        sb.stat.strikeOuts +
        " Ks. " +
        (parseFloat(sa.stat.era) < parseFloat(sb.stat.era) ? a.name : b.name) +
        " has the better ERA right now. =)"
      );
    }
    var winner = parseFloat(sa.stat.ops) >= parseFloat(sb.stat.ops) ? a.name : b.name;
    return (
      a.name +
      " vs " +
      b.name +
      " " +
      sa.label +
      ": " +
      a.name +
      " is " +
      sa.stat.avg +
      "/" +
      sa.stat.obp +
      "/" +
      sa.stat.slg +
      " (" +
      sa.stat.ops +
      " OPS, " +
      sa.stat.homeRuns +
      " HR); " +
      b.name +
      " is " +
      sb.stat.avg +
      "/" +
      sb.stat.obp +
      "/" +
      sb.stat.slg +
      " (" +
      sb.stat.ops +
      " OPS, " +
      sb.stat.homeRuns +
      " HR). Edge in OPS: " +
      winner +
      ". Want ERA instead, or last year? =)"
    );
  }

  function lastName(full) {
    var parts = String(full || "").trim().split(/\s+/);
    if (!parts.length) return "";
    if (parts.length > 1 && /^jr\.?$/i.test(parts[parts.length - 1])) {
      return parts.slice(-2).join(" ");
    }
    return parts[parts.length - 1];
  }

  function occupied(base) {
    return !!(base && (base === true || base.id || base.fullName));
  }

  function basesText(off) {
    if (!off) return "Bases empty.";
    var bits = [];
    if (occupied(off.first)) bits.push("first");
    if (occupied(off.second)) bits.push("second");
    if (occupied(off.third)) bits.push("third");
    if (!bits.length) return "Bases empty.";
    if (bits.length === 3) return "Bases loaded.";
    if (bits.length === 1) return "Runner on " + bits[0] + ".";
    return "Runners on " + bits.join(" and ") + ".";
  }

  function isLiveGame(g) {
    var s = (g && g.status && (g.status.detailedState || g.status.abstractGameState)) || "";
    return /progress|live/i.test(s);
  }

  function fieldBoard(g, extra) {
    if (!g || !isLiveGame(g)) return null;
    var away = g.teams && g.teams.away;
    var home = g.teams && g.teams.home;
    if (!away || !home) return null;
    var ls = (extra && extra.linescore) || g.linescore || {};
    var off = ls.offense || {};
    var def = ls.defense || {};
    return {
      kind: "mlb-field",
      live: true,
      away: (away.team && (away.team.teamName || away.team.name)) || "Away",
      home: (home.team && (home.team.teamName || home.team.name)) || "Home",
      awayScore: away.score != null ? away.score : "",
      homeScore: home.score != null ? home.score : "",
      inningState: ls.inningState || "",
      inning: ls.currentInningOrdinal || "",
      balls: ls.balls != null ? Number(ls.balls) : 0,
      strikes: ls.strikes != null ? Number(ls.strikes) : 0,
      outs: ls.outs != null ? Number(ls.outs) : 0,
      batter: (off.batter && off.batter.fullName) || "",
      pitcher: (def.pitcher && def.pitcher.fullName) || "",
      onDeck: (off.onDeck && off.onDeck.fullName) || "",
      first: occupied(off.first),
      second: occupied(off.second),
      third: occupied(off.third),
    };
  }

  function rememberBoard(g, extra) {
    ctx.board = fieldBoard(g, extra);
  }

  function formatGame(g, teamFilter, compact, extra) {
    if (!g) return "";
    var away = g.teams && g.teams.away;
    var home = g.teams && g.teams.home;
    if (!away || !home) return "";
    var an = (away.team && (away.team.teamName || away.team.name)) || "Away";
    var hn = (home.team && (home.team.teamName || home.team.name)) || "Home";
    if (teamFilter) {
      var blob = norm(
        an + " " + hn + " " + ((away.team && away.team.name) || "") + " " + ((home.team && home.team.name) || "")
      );
      if (blob.indexOf(norm(teamFilter)) === -1) return "";
    }
    var state = (g.status && g.status.detailedState) || "";
    var ls = (extra && extra.linescore) || g.linescore || {};
    var as = away.score != null ? away.score : "";
    var hs = home.score != null ? home.score : "";
      if (/progress|live/i.test(state) || isLiveGame(g)) {
      var core = an + " " + as + ", " + hn + " " + hs + " — " + (ls.inningState || "") + " " + (ls.currentInningOrdinal || "");
      var outs = Number(ls.outs);
      if (outs >= 3) {
        return (core + ", inning break.").replace(/\s+/g, " ");
      }
      var outTxt = outs === 1 ? "1 out" : (isNaN(outs) ? "" : outs + " outs");
      var count =
        ls.balls != null && ls.strikes != null ? ls.balls + "-" + ls.strikes + " count" : "";
      var pitcher = ls.defense && ls.defense.pitcher && ls.defense.pitcher.fullName;
      var batter = ls.offense && ls.offense.batter && ls.offense.batter.fullName;
      var deck = ls.offense && ls.offense.onDeck && ls.offense.onDeck.fullName;
      if (compact) {
        return [core, outTxt, count, pitcher && batter ? lastName(pitcher) + " to " + lastName(batter) : ""]
          .filter(Boolean)
          .join(", ")
          .replace(/\s+/g, " ") + ".";
      }
      var bits = [core + (outTxt || count ? "," : ".")];
      if (outTxt) bits.push(outTxt + (count ? "," : "."));
      if (count) bits.push(count + ".");
      if (pitcher && batter) bits.push(pitcher + " pitching to " + batter + ".");
      else if (pitcher) bits.push(pitcher + " on the mound.");
      else if (batter) bits.push(batter + " at the plate.");
      if (deck) bits.push("On deck: " + deck + ".");
      bits.push(basesText(ls.offense));
      if (extra && extra.description) bits.push(extra.description + (/\.$/.test(extra.description) ? "" : "."));
      else if (extra && extra.pitch) bits.push("Last pitch: " + extra.pitch + ".");
      return bits.join(" ").replace(/\s+/g, " ").replace(/\s+\./g, ".");
    }
    if (/final/i.test(state)) {
      var aw = parseInt(as, 10);
      var hw = parseInt(hs, 10);
      if (aw > hw) return an + " beat the " + hn + " " + as + "-" + hs + ".";
      if (hw > aw) return hn + " beat the " + an + " " + hs + "-" + as + ".";
      return an + " and " + hn + " finished " + as + "-" + hs + ".";
    }
    var ap = away.probablePitcher && away.probablePitcher.fullName;
    var hp = home.probablePitcher && home.probablePitcher.fullName;
    var when = an + " at " + hn + " — " + (state || "scheduled") + ".";
    if (ap || hp) when += " Probables: " + (ap || "?") + " vs " + (hp || "?") + ".";
    return when;
  }

  async function mlbLiveBits(gamePk) {
    if (!gamePk) return null;
    try {
      var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
      var t = setTimeout(function () {
        if (ctrl) ctrl.abort();
      }, 8000);
      var res = await fetch("https://statsapi.mlb.com/api/v1.1/game/" + gamePk + "/feed/live", {
        signal: ctrl ? ctrl.signal : undefined,
      });
      clearTimeout(t);
      if (!res.ok) return null;
      var data = await res.json();
      var cur = (((data.liveData || {}).plays || {}).currentPlay) || {};
      var events = cur.playEvents || [];
      var last = events.length ? events[events.length - 1] : null;
      return {
        description: (cur.result && cur.result.description) || "",
        pitch: last && last.details && last.details.description,
        linescore: (data.liveData || {}).linescore || null,
      };
    } catch (err) {
      return null;
    }
  }

  async function scores(q) {
    if (q.league && q.league !== "mlb") return espnScores(q);
    ctx.lastLeague = "mlb";
    ctx.espnClub = null;
    var iso = todayISO();
    var data = await mlb("/schedule?sportId=1&date=" + iso + "&hydrate=team,linescore(matchup,runners),probablePitcher");
    var games = ((data.dates || [])[0] && data.dates[0].games) || [];
    if (!games.length) return "No MLB games on the slate I see for today. I can still pull a player’s season. =)";
    var live = [];
    games.forEach(function (g) {
      if (isLiveGame(g)) live.push(g);
    });
    if (q.team) {
      var picked = null;
      games.forEach(function (g) {
        if (!picked && formatGame(g, q.team)) picked = g;
      });
      ctx.team = q.team;
      if (!picked) return "I don’t see the " + q.team + " on today’s MLB board. Want standings instead? =)";
      ctx.lastGamePk = picked.gamePk || ctx.lastGamePk;
      var extra = isLiveGame(picked) ? await mlbLiveBits(picked.gamePk) : null;
      rememberBoard(picked, extra);
      var line = formatGame(picked, q.team, false, extra);
      return line + " Ask who’s up, the count, or another club. =)";
    }
    if (liveAsk(q.low)) {
      var focused = null;
      if (ctx.lastGamePk) {
        games.forEach(function (g) {
          if (g.gamePk === ctx.lastGamePk) focused = g;
        });
      }
      if (!focused) focused = live[0] || games[0];
      if (focused) {
        ctx.lastGamePk = focused.gamePk || ctx.lastGamePk;
        var bits = isLiveGame(focused) ? await mlbLiveBits(focused.gamePk) : null;
        rememberBoard(focused, bits);
        return (
          formatGame(focused, null, false, bits) +
          " Name a club if you meant a different game. =)"
        );
      }
    }
    var shown = (live.length ? live : games).slice(0, 4);
    ctx.lastGamePk = (shown[0] && shown[0].gamePk) || ctx.lastGamePk;
    if (live[0]) rememberBoard(live[0], null);
    var head = shown
      .map(function (g) {
        return formatGame(g, null, true);
      })
      .filter(Boolean)
      .join(" ");
    return (
      "MLB today: " +
      games.length +
      " games" +
      (live.length ? ", " + live.length + " live" : "") +
      ". " +
      head +
      " Name a club — Cubs, Dodgers — for the full count and matchup. =)"
    );
  }

  async function standings(q) {
    if (q.league && q.league !== "mlb") {
      var spec = ESPN_BOARDS[q.league];
      return (
        "Division tables are still MLB-first. Say “" +
        ((spec && spec.label) || q.league) +
        " scores” and I’ll read the live board. =)"
      );
    }
    var year = seasonYear();
    var data = await mlb("/standings?leagueId=103,104&season=" + year + "&standingsTypes=regularSeason");
    var recs = data.records || [];
    function lines(rec) {
      var div = DIVISIONS[rec.division && rec.division.id] || "Division";
      var rows = (rec.teamRecords || []).map(function (t, i) {
        var name = (t.team && (t.team.teamName || t.team.name)) || "Team";
        return name + " " + t.wins + "-" + t.losses + (i === 0 ? " (1st)" : "");
      });
      return div + ": " + rows.join(", ") + ".";
    }
    if (q.team) {
      var hit = "";
      recs.forEach(function (rec) {
        var names = (rec.teamRecords || []).map(function (t) {
          return norm((t.team && t.team.name) || "") + " " + norm((t.team && t.team.teamName) || "");
        }).join(" ");
        if (names.indexOf(norm(q.team)) !== -1) hit = lines(rec);
      });
      return hit || "I couldn’t place that club in the live table. Try Cubs or AL East. =)";
    }
    var firsts = recs.map(function (rec) {
      var div = DIVISIONS[rec.division && rec.division.id] || "Division";
      var t = (rec.teamRecords || [])[0];
      if (!t) return "";
      return div + " " + ((t.team && (t.team.teamName || t.team.name)) || "") + " " + t.wins + "-" + t.losses;
    }).filter(Boolean);
    return "MLB first place right now: " + firsts.join("; ") + ". Name a division or a team for the full stack. =)";
  }

  async function leaders(q) {
    var year = seasonYear();
    var cat = "homeRuns";
    var label = "home runs";
    if (/\bera\b/.test(q.low)) {
      cat = "earnedRunAverage";
      label = "ERA";
    } else if (/\bops\b/.test(q.low)) {
      cat = "onBasePlusSlugging";
      label = "OPS";
    } else if (/\baverage|avg\b/.test(q.low)) {
      cat = "battingAverage";
      label = "batting average";
    } else if (/\brbi/.test(q.low)) {
      cat = "runsBattedIn";
      label = "RBIs";
    } else if (/\bstrikeout/.test(q.low)) {
      cat = "strikeouts";
      label = "strikeouts";
    }
    var data = await mlb("/stats/leaders?leaderCategories=" + cat + "&season=" + year + "&sportId=1&limit=5");
    var pack = (data.leagueLeaders || [])[0];
    var list = (pack && pack.leaders) || [];
    if (!list.length) return "The live leaderboard came back empty. Try a player name instead. =)";
    var bits = list.slice(0, 5).map(function (row) {
      return (row.rank || "") + ". " + ((row.person && row.person.fullName) || "?") + " " + row.value;
    });
    if (list[0] && list[0].person) {
      var person = await findPlayer(list[0].person.fullName);
      if (person) ctx.player = await loadCard(person);
    }
    return "MLB " + label + " leaders this season: " + bits.join("; ") + ". Want that first name’s full page? =)";
  }

  async function espnBoard(league) {
    var spec = ESPN_BOARDS[league];
    if (!spec) return null;
    var hit = cache.espn[league];
    if (hit && Date.now() - hit.at < 45 * 1000) return hit.data;
    var data = await espn(spec.path);
    cache.espn[league] = { at: Date.now(), data: data };
    return data;
  }

  function espnEventBlob(ev) {
    var comp = (ev.competitions || [])[0] || {};
    var names = (comp.competitors || []).map(function (c) {
      var t = c.team || {};
      return [t.displayName, t.shortDisplayName, t.abbreviation, t.nickname, t.location].join(" ");
    });
    return norm(names.join(" ") + " " + (ev.shortName || "") + " " + (ev.name || ""));
  }

  function espnSides(ev) {
    var comp = (ev.competitions || [])[0] || {};
    var away = { name: "Away", score: "", id: "" };
    var home = { name: "Home", score: "", id: "" };
    (comp.competitors || []).forEach(function (c) {
      var t = c.team || {};
      var row = {
        name: t.displayName || t.shortDisplayName || t.abbreviation || "Team",
        short: t.abbreviation || t.shortDisplayName || t.displayName || "Team",
        score: c.score != null && c.score !== "" ? c.score : "0",
        id: String(c.id || (t && t.id) || ""),
        winner: !!c.winner,
      };
      if (c.homeAway === "home") home = row;
      else away = row;
    });
    return { away: away, home: home, comp: comp };
  }

  function espnPossName(comp, sit) {
    if (!sit || sit.possession == null) return "";
    var pid = String(sit.possession);
    var found = "";
    (comp.competitors || []).forEach(function (c) {
      var tid = String(c.id || (c.team && c.team.id) || "");
      if (tid === pid) found = (c.team && (c.team.abbreviation || c.team.displayName)) || "";
    });
    return found;
  }

  function formatEspnEvent(ev, needle, compact) {
    if (!ev) return "";
    if (needle && espnEventBlob(ev).indexOf(norm(needle)) === -1) return "";
    var sides = espnSides(ev);
    var st = (ev.status && ev.status.type) || {};
    var state = st.state || "";
    var detail = st.detail || st.shortDetail || "";
    var sit = (sides.comp && sides.comp.situation) || {};
    var score = sides.away.short + " " + sides.away.score + ", " + sides.home.short + " " + sides.home.score;
    if (state === "in") {
      var bits = [score + " — " + (detail || "live")];
      if (sit.downDistanceText) bits.push(sit.downDistanceText);
      var poss = espnPossName(sides.comp, sit);
      if (poss) bits.push(poss + " have it");
      if (sit.possessionText && !sit.downDistanceText) bits.push(sit.possessionText);
      var last = sit.lastPlay && sit.lastPlay.text;
      if (last && !compact) {
        if (last.length > 110) last = last.slice(0, 107) + "…";
        bits.push("Last: " + String(last).replace(/[.]+$/, ""));
      }
      var liveLine = bits.filter(Boolean).join(". ");
      if (!/[.!?…]$/.test(liveLine)) liveLine += ".";
      return liveLine.replace(/\.{2,}/g, ".");
    }
    if (state === "post") {
      var winner = sides.away.winner ? sides.away : sides.home.winner ? sides.home : null;
      var loser = sides.away.winner ? sides.home : sides.home.winner ? sides.away : null;
      if (winner && loser) return winner.name + " " + winner.score + "-" + loser.score + " over " + loser.name + ".";
      return sides.away.name + " " + sides.away.score + ", " + sides.home.name + " " + sides.home.score + " — " + (detail || "final") + ".";
    }
    return sides.away.name + " at " + sides.home.name + " — " + (detail || "scheduled") + ".";
  }

  function espnStateRank(ev) {
    var state = (ev.status && ev.status.type && ev.status.type.state) || "";
    if (state === "in") return 0;
    if (state === "post") return 1;
    return 2;
  }

  async function espnScores(q) {
    var league = q.league && ESPN_BOARDS[q.league] ? q.league : "nba";
    var spec = ESPN_BOARDS[league];
    var data = await espnBoard(league);
    var events = (data && data.events) || [];
    ctx.lastLeague = league;
    ctx.team = null;
    var club = q.espnClub || findEspnClub(q.low);
    if (club) ctx.espnClub = club;
    if (!events.length) {
      return "I don’t see a " + spec.label + " board right now. Try a player name instead. =)";
    }
    if (club) {
      var hit = "";
      events.forEach(function (ev) {
        if (!hit) hit = formatEspnEvent(ev, club.needle, false);
      });
      return hit
        ? hit + " I can check another " + spec.label + " club, or a player. =)"
        : "I don’t see " +
            club.needle +
            " on today’s " +
            spec.label +
            " board. " +
            (events[0] ? "Next I do see: " + formatEspnEvent(events[0], "", true) : "") +
            " =)";
    }
    var ranked = events.slice().sort(function (a, b) {
      return espnStateRank(a) - espnStateRank(b);
    });
    var liveN = events.filter(function (ev) {
      return espnStateRank(ev) === 0;
    }).length;
    var shown = ranked.slice(0, liveN ? Math.min(5, Math.max(liveN, 1)) : 4);
    var head = shown
      .map(function (ev) {
        return formatEspnEvent(ev, "", true);
      })
      .filter(Boolean)
      .join(" ");
    return (
      spec.label +
      ": " +
      events.length +
      " games" +
      (liveN ? ", " + liveN + " live" : "") +
      ". " +
      head +
      " Name a club for one board. =)"
    );
  }

  function espnAthleteId(item) {
    var uid = String((item && item.uid) || "");
    var m = uid.match(/a:(\d+)/);
    if (m) return m[1];
    var web = (item && item.link && item.link.web) || "";
    m = String(web).match(/\/id\/(\d+)/);
    return m ? m[1] : "";
  }

  function leagueFromEspnItem(item) {
    var slug = String((item && item.defaultLeagueSlug) || "").toLowerCase();
    var k;
    for (k in ESPN_BOARDS) {
      if (Object.prototype.hasOwnProperty.call(ESPN_BOARDS, k) && ESPN_BOARDS[k].slug === slug) return k;
    }
    var sport = String((item && item.sport) || "").toLowerCase();
    if (sport === "basketball") return "nba";
    if (sport === "football") return "nfl";
    if (sport === "hockey") return "nhl";
    if (sport === "soccer") return "epl";
    return "";
  }

  function pickEspnSplit(stats, window) {
    var splits = (stats && stats.splits) || [];
    if (!splits.length) return null;
    var wantCareer = window === "career";
    var i;
    var row;
    for (i = 0; i < splits.length; i++) {
      row = splits[i];
      var name = String(row.displayName || "").toLowerCase();
      if (wantCareer && /career/.test(name)) return row;
      if (!wantCareer && /regular/.test(name)) return row;
    }
    if (wantCareer) return splits[splits.length - 1];
    return splits[0];
  }

  function espnSplitLine(stats, split) {
    if (!split || !split.stats) return "";
    var names = (stats && stats.names) || [];
    var labels = (stats && stats.labels) || [];
    var display = (stats && stats.displayNames) || [];
    var map = {};
    function put(k, v) {
      if (k == null || v == null || v === "") return;
      map[String(k)] = v;
      map[String(k).toUpperCase()] = v;
    }
    names.forEach(function (k, i) {
      put(k, split.stats[i]);
    });
    labels.forEach(function (k, i) {
      put(k, split.stats[i]);
    });
    display.forEach(function (k, i) {
      put(k, split.stats[i]);
    });
    var bits = [];
    if (map.PTS || map.points || map.avgPoints) bits.push((map.PTS || map.points || map.avgPoints) + " points");
    if (map.REB || map.rebounds || map.avgRebounds) bits.push((map.REB || map.rebounds || map.avgRebounds) + " boards");
    if (map.AST || map.assists || map.avgAssists) bits.push((map.AST || map.assists || map.avgAssists) + " assists");
    if (map.passingYards) bits.push(map.passingYards + " passing yards");
    if (map.passingTouchdowns) bits.push(map.passingTouchdowns + " passing TDs");
    if (map.QBRating) bits.push(map.QBRating + " rating");
    if (map.goals) bits.push(map.goals + " goals");
    if ((map.A || map.assists) && !(map.PTS || map.points || map.AST || map.avgAssists)) {
      bits.push((map.A || map.assists) + " assists");
    }
    if (map.P && !(map.PTS || map.points)) bits.push(map.P + " points");
    if (map.GP || map.gamesPlayed) bits.push("in " + (map.GP || map.gamesPlayed) + " games");
    if (!bits.length) {
      var shown = (labels.length ? labels : names).slice(0, 5);
      bits = shown.map(function (lab, i) {
        return split.stats[i] + " " + lab;
      });
    }
    return bits.join(", ");
  }

  function formatEspnAthlete(card, q) {
    if (!card) return "That player page came back empty. =)";
    var split = pickEspnSplit(card.stats, q.window);
    var line = espnSplitLine(card.stats, split);
    var label = (split && split.displayName) || (q.window === "career" ? "career" : "this season");
    ctx.espnPlayer = card;
    ctx.lastLeague = card.league || ctx.lastLeague;
    ctx.player = null;
    if (!line) {
      return "I found " + card.name + " in the " + (card.label || "league") + ", but that stat split is empty. Try career. =)";
    }
    return (
      card.name +
      " (" +
      (card.label || card.league) +
      ", " +
      label +
      "): " +
      line +
      ". Want the other split, or the live " +
      (card.label || "league") +
      " board? =)"
    );
  }

  async function espnPlayer(q) {
    var name = q.kind === "follow" && ctx.espnPlayer ? ctx.espnPlayer.name : q.names && q.names[0];
    if (!name) return null;
    var data = await espn("/search/v2?region=us&lang=en&limit=8&query=" + encodeURIComponent(name));
    var players = [];
    (data.results || []).forEach(function (block) {
      (block.contents || []).forEach(function (item) {
        if (item && item.type === "player") players.push(item);
      });
    });
    if (!players.length) return null;
    var want = q.league && q.league !== "mlb" ? q.league : "";
    var item = players[0];
    var i;
    for (i = 0; i < players.length; i++) {
      var lg = leagueFromEspnItem(players[i]);
      if (want && lg === want) {
        item = players[i];
        break;
      }
      if (!want && lg && lg !== "mlb") {
        item = players[i];
        break;
      }
    }
    var id = espnAthleteId(item);
    var league = leagueFromEspnItem(item) || want || "nba";
    var spec = ESPN_BOARDS[league];
    if (!id || !spec) return null;
    var overview = await espn("/common/v3/sports/" + spec.sport + "/" + spec.slug + "/athletes/" + id + "/overview");
    var card = {
      id: id,
      name: item.displayName || name,
      league: league,
      label: spec.label,
      stats: overview.statistics || {},
    };
    return formatEspnAthlete(card, q);
  }

  function otherSportsReply(q) {
    var league = (q.low.match(UNSUPPORTED_RE) || q.low.match(OTHER_RE) || ["that sport"])[0];
    return (
      "I don’t have a live " +
      league +
      " feed in this beta. I can do MLB boards (count, pitcher, hitter) plus NBA, NFL, NHL, WNBA, college football, MLS, and Premier League scores. =)"
    );
  }

  async function answer(text) {
    try {
      var q = parse(text);
      ctx.lastAsk = q.raw;
      ctx.board = null;
      if (q.kind === "other") return otherSportsReply(q);
      if (q.kind === "scores") return scores(q);
      if (q.kind === "standings") return standings(q);
      if (q.kind === "leaders") {
        if (q.league && q.league !== "mlb") {
          return "Leaderboards are still MLB-first. Name a player and I’ll pull their " + (ESPN_BOARDS[q.league] && ESPN_BOARDS[q.league].label) + " page. =)";
        }
        return leaders(q);
      }
      if (q.kind === "compare") {
        if (q.league && q.league !== "mlb") {
          q.kind = "player";
          return talkPlayer(q);
        }
        return compare(q);
      }
      return talkPlayer(q);
    } catch (err) {
      ctx.board = null;
      return "Live scoreboard unavailable right now. Try again in a moment. =)";
    }
  }

  function reset() {
    ctx = {
      player: null,
      team: null,
      lastStat: null,
      lastAsk: "",
      lastLeague: "mlb",
      lastGamePk: null,
      espnPlayer: null,
      espnClub: null,
      board: null,
    };
  }

  var api = {
    looksSports: looksSports,
    parse: parse,
    extractNames: extractNames,
    findTeam: findTeam,
    detectLeague: detectLeague,
    formatGame: formatGame,
    fieldBoard: fieldBoard,
    seasonYear: seasonYear,
    commentOPS: commentOPS,
    commentERA: commentERA,
    commentAVG: commentAVG,
    answer: answer,
    reset: reset,
    clearBoard: function () {
      ctx.board = null;
    },
    get board() {
      return ctx.board;
    },
    get context() {
      return ctx;
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PyxAssistantSports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
