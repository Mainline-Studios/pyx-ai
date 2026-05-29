/**
 * Pyx Account — Firebase Auth (Google + email/password + email magic-link) and
 * Firestore-backed chat sync + sharing. Loaded as an ES module; exposes a global
 * `window.PyxAccount` with a `ready` promise.
 *
 * Data model:
 *   users/{uid}/chats/{chatId}  -> { title, line, messages[], updatedAt, createdAt }
 *   shared/{shareId}            -> { ownerUid, title, line, messages[], createdAt }
 *
 * Security rules (firestore.rules): users/{uid}/** is owner-only; shared/{id} is
 * world-readable but owner-writable. Viewers who reply fork a copy into their own
 * users/{uid}/chats.
 */
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getAuth, setPersistence, browserLocalPersistence, onAuthStateChanged,
  GoogleAuthProvider, signInWithPopup,
  signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut,
  sendSignInLinkToEmail, isSignInWithEmailLink, signInWithEmailLink,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";
import {
  getFirestore, collection, doc, setDoc, deleteDoc, getDoc, getDocs,
  onSnapshot, query, orderBy, addDoc, serverTimestamp,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyB3XVQa49aLpOfRMXI0XNIM_pmP_DisM7A",
  authDomain: "pyx-ai.firebaseapp.com",
  projectId: "pyx-ai",
  storageBucket: "pyx-ai.firebasestorage.app",
  messagingSenderId: "574247481583",
  appId: "1:574247481583:web:366a9f3ed9b2e6d4036b1a",
  measurementId: "G-XW1Z8ZWKCK",
};

const EMAIL_LINK_KEY = "pyx.account.emailForSignIn";

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

let currentUser = null;
const authListeners = new Set();

function publicUser(u) {
  if (!u) return null;
  return {
    uid: u.uid,
    email: u.email || "",
    displayName: u.displayName || "",
    photoURL: u.photoURL || "",
  };
}

const ready = setPersistence(auth, browserLocalPersistence)
  .catch(function () {})
  .then(function () {
    return new Promise(function (resolve) {
      let resolved = false;
      onAuthStateChanged(auth, function (u) {
        currentUser = u;
        const pub = publicUser(u);
        authListeners.forEach(function (cb) {
          try { cb(pub); } catch (e) {}
        });
        if (!resolved) { resolved = true; resolve(pub); }
      });
    });
  });

function requireUser() {
  if (!currentUser) throw new Error("Not signed in");
  return currentUser;
}

function chatsCol(uid) {
  return collection(db, "users", uid, "chats");
}

const PyxAccount = {
  ready: ready,

  currentUser: function () { return publicUser(currentUser); },

  onAuth: function (cb) {
    authListeners.add(cb);
    // Fire immediately with the latest known state.
    try { cb(publicUser(currentUser)); } catch (e) {}
    return function () { authListeners.delete(cb); };
  },

  signInGoogle: function () {
    const provider = new GoogleAuthProvider();
    provider.setCustomParameters({ prompt: "select_account" });
    return signInWithPopup(auth, provider).then(function (res) { return publicUser(res.user); });
  },

  signInEmailPassword: function (email, password) {
    return signInWithEmailAndPassword(auth, String(email).trim(), password)
      .then(function (res) { return publicUser(res.user); });
  },

  registerEmailPassword: function (email, password) {
    return createUserWithEmailAndPassword(auth, String(email).trim(), password)
      .then(function (res) { return publicUser(res.user); });
  },

  sendMagicLink: function (email, continueUrl) {
    const e = String(email).trim();
    return sendSignInLinkToEmail(auth, e, { url: continueUrl, handleCodeInApp: true })
      .then(function () {
        try { localStorage.setItem(EMAIL_LINK_KEY, e); } catch (er) {}
        return true;
      });
  },

  isMagicLink: function () {
    return isSignInWithEmailLink(auth, window.location.href);
  },

  completeMagicLink: function (promptEmail) {
    if (!isSignInWithEmailLink(auth, window.location.href)) return Promise.resolve(null);
    let email = "";
    try { email = localStorage.getItem(EMAIL_LINK_KEY) || ""; } catch (e) {}
    if (!email && typeof promptEmail === "function") email = promptEmail() || "";
    if (!email) return Promise.reject(new Error("Email required to finish sign-in"));
    return signInWithEmailLink(auth, email, window.location.href).then(function (res) {
      try { localStorage.removeItem(EMAIL_LINK_KEY); } catch (e) {}
      return publicUser(res.user);
    });
  },

  signOutUser: function () { return signOut(auth); },

  // ---- Chats (require sign-in) ----

  watchChats: function (cb) {
    const u = requireUser();
    const q = query(chatsCol(u.uid), orderBy("updatedAt", "desc"));
    return onSnapshot(q, function (snap) {
      const out = [];
      snap.forEach(function (d) {
        const data = d.data() || {};
        out.push({
          id: d.id,
          title: data.title || "Chat",
          line: data.line === "preview" ? "preview" : "cloud",
          messages: Array.isArray(data.messages) ? data.messages : [],
          updatedAt: data.updatedAt && data.updatedAt.toMillis ? data.updatedAt.toMillis() : 0,
        });
      });
      try { cb(out); } catch (e) {}
    }, function () {});
  },

  getChats: function () {
    const u = requireUser();
    return getDocs(query(chatsCol(u.uid), orderBy("updatedAt", "desc"))).then(function (snap) {
      const out = [];
      snap.forEach(function (d) {
        const data = d.data() || {};
        out.push({
          id: d.id,
          title: data.title || "Chat",
          line: data.line === "preview" ? "preview" : "cloud",
          messages: Array.isArray(data.messages) ? data.messages : [],
          updatedAt: data.updatedAt && data.updatedAt.toMillis ? data.updatedAt.toMillis() : 0,
        });
      });
      return out;
    });
  },

  saveChat: function (chat) {
    const u = requireUser();
    if (!chat || !chat.id) return Promise.reject(new Error("chat.id required"));
    const payload = {
      title: String(chat.title || "Chat").slice(0, 120),
      line: chat.line === "preview" ? "preview" : "cloud",
      messages: Array.isArray(chat.messages) ? chat.messages : [],
      updatedAt: serverTimestamp(),
    };
    return setDoc(doc(db, "users", u.uid, "chats", String(chat.id)), payload, { merge: true });
  },

  deleteChat: function (chatId) {
    const u = requireUser();
    return deleteDoc(doc(db, "users", u.uid, "chats", String(chatId)));
  },

  // ---- Sharing ----

  shareChat: function (chat) {
    const u = requireUser();
    const payload = {
      ownerUid: u.uid,
      ownerName: u.displayName || u.email || "Someone",
      title: String((chat && chat.title) || "Shared chat").slice(0, 120),
      line: chat && chat.line === "preview" ? "preview" : "cloud",
      messages: Array.isArray(chat && chat.messages) ? chat.messages : [],
      createdAt: serverTimestamp(),
    };
    return addDoc(collection(db, "shared"), payload).then(function (ref) {
      return {
        id: ref.id,
        url: window.location.origin + "/pyx-talk.html?shared=" + encodeURIComponent(ref.id),
      };
    });
  },

  getShared: function (shareId) {
    return getDoc(doc(db, "shared", String(shareId))).then(function (snap) {
      if (!snap.exists()) return null;
      const data = snap.data() || {};
      return {
        id: snap.id,
        ownerUid: data.ownerUid || "",
        ownerName: data.ownerName || "Someone",
        title: data.title || "Shared chat",
        line: data.line === "preview" ? "preview" : "cloud",
        messages: Array.isArray(data.messages) ? data.messages : [],
      };
    });
  },
};

window.PyxAccount = PyxAccount;
window.dispatchEvent(new Event("pyx-account-ready"));
