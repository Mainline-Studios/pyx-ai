/**
 * Pyx Assistant — live weather via Open-Meteo (no API key).
 */
(function (root) {
  "use strict";

  var CODES = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
  };

  function extractPlace(text) {
    var t = String(text || "");
    var m =
      t.match(/\b(?:weather|forecast|temperature)\s+(?:in|for|at)\s+(.+)$/i) ||
      t.match(/\b(?:in|for|at)\s+([a-zA-Z][a-zA-Z\s,'-]{1,40})$/i);
    if (!m) return "";
    return m[1]
      .replace(/[?.!]+$/g, "")
      .replace(/\b(today|now|please|right now)\b/gi, "")
      .trim();
  }

  async function geocode(place) {
    var url =
      "https://geocoding-api.open-meteo.com/v1/search?name=" +
      encodeURIComponent(place) +
      "&count=1&language=en&format=json";
    var res = await fetch(url);
    if (!res.ok) throw new Error("geocode " + res.status);
    var data = await res.json();
    var hit = data && data.results && data.results[0];
    if (!hit) return null;
    return {
      name: hit.name,
      admin: hit.admin1 || "",
      country: hit.country_code || hit.country || "",
      lat: hit.latitude,
      lon: hit.longitude,
    };
  }

  async function forecast(lat, lon) {
    var url =
      "https://api.open-meteo.com/v1/forecast?latitude=" +
      encodeURIComponent(lat) +
      "&longitude=" +
      encodeURIComponent(lon) +
      "&current=temperature_2m,weather_code,wind_speed_10m&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto";
    var res = await fetch(url);
    if (!res.ok) throw new Error("forecast " + res.status);
    return res.json();
  }

  async function answer(text) {
    var place = extractPlace(text);
    if (!place) {
      return "Tell me a city — try “weather in austin” — and I’ll pull a live snapshot.";
    }
    try {
      var geo = await geocode(place);
      if (!geo) return "I couldn’t find “" + place + ".” Try a bigger city name.";
      var data = await forecast(geo.lat, geo.lon);
      var cur = data && data.current;
      if (!cur) return "Weather feed came back empty. Try again in a moment.";
      var label = CODES[cur.weather_code] || "mixed conditions";
      var where = geo.name + (geo.admin ? ", " + geo.admin : "") + (geo.country ? " (" + geo.country + ")" : "");
      var temp = Math.round(Number(cur.temperature_2m));
      var wind = Math.round(Number(cur.wind_speed_10m));
      return (
        "Live in " +
        where +
        ": " +
        temp +
        "°F, " +
        label +
        ", wind about " +
        wind +
        " mph. (Open-Meteo snapshot — not a full forecast.)"
      );
    } catch (err) {
      return "I couldn’t reach live weather just now. Try again in a second, or convert temperatures with me locally.";
    }
  }

  root.PyxAssistantWeather = {
    extractPlace: extractPlace,
    answer: answer,
  };
})(typeof window !== "undefined" ? window : globalThis);
