/* Minimal promise wrapper over IndexedDB for the offline layer: pack
   manifests + the audit sync queue. One database, two object stores.
   Everything degrades to "unavailable" (null / empty) when IndexedDB is
   missing (SSR, locked-down browsers) so callers never crash. */

export const DB_NAME = "asvproject-offline";
export const DB_VERSION = 1;
export const STORE_PACKS = "packs";
export const STORE_QUEUE = "auditQueue";

function hasIdb(): boolean {
  return typeof indexedDB !== "undefined";
}

let dbPromise: Promise<IDBDatabase> | null = null;
let dbInstance: IDBDatabase | null = null;

export function openDb(): Promise<IDBDatabase> {
  if (!hasIdb()) return Promise.reject(new Error("IndexedDB unavailable"));
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_PACKS)) {
        db.createObjectStore(STORE_PACKS, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(STORE_QUEUE)) {
        const q = db.createObjectStore(STORE_QUEUE, { keyPath: "id" });
        q.createIndex("clip_key", "clip_key", { unique: false });
      }
    };
    req.onsuccess = () => {
      dbInstance = req.result;
      // another tab upgrading/deleting the DB: drop our handle so the next
      // call reopens instead of using a closed connection
      dbInstance.onversionchange = () => {
        dbInstance?.close();
        dbInstance = null;
        dbPromise = null;
      };
      resolve(req.result);
    };
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
    req.onblocked = () => reject(new Error("IndexedDB open blocked"));
  });
  dbPromise.catch(() => {
    dbPromise = null;
  });
  return dbPromise;
}

function reqToPromise<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"));
  });
}

export async function idbGetAll<T>(store: string): Promise<T[]> {
  try {
    const db = await openDb();
    const tx = db.transaction(store, "readonly");
    return (await reqToPromise(tx.objectStore(store).getAll())) as T[];
  } catch {
    return [];
  }
}

export async function idbGet<T>(store: string, key: string): Promise<T | null> {
  try {
    const db = await openDb();
    const tx = db.transaction(store, "readonly");
    return ((await reqToPromise(tx.objectStore(store).get(key))) as T | undefined) ?? null;
  } catch {
    return null;
  }
}

export async function idbPut<T>(store: string, value: T): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(store, "readwrite");
  await reqToPromise(tx.objectStore(store).put(value));
}

export async function idbDelete(store: string, key: string): Promise<void> {
  try {
    const db = await openDb();
    const tx = db.transaction(store, "readwrite");
    await reqToPromise(tx.objectStore(store).delete(key));
  } catch {
    /* nothing to delete */
  }
}

/** Test hook: drop the whole database (jsdom/fake-indexeddb). */
export async function idbReset(): Promise<void> {
  if (!hasIdb()) return;
  // an open connection blocks deleteDatabase forever: close it first
  dbInstance?.close();
  dbInstance = null;
  dbPromise = null;
  await new Promise<void>((resolve) => {
    const req = indexedDB.deleteDatabase(DB_NAME);
    req.onsuccess = () => resolve();
    req.onerror = () => resolve();
    req.onblocked = () => resolve();
  });
}
