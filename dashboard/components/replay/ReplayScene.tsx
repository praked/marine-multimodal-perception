"use client";

import {
  horizontalToVector,
  moonPosition,
  sunPosition,
  type HorizontalPos,
} from "@/lib/celestial";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/* 3D mission replay: satellite ground of the capture sector, the boat at
   the GPS position, the sun (NASA model) and moon placed by their REAL
   ephemeris for the scrubbed instant, lighting to match, and the current
   radar returns floating in the boat frame.

   Scene frame: X = east, Y = up, Z = south (north = -Z). Metres. */

const SECTOR_M = 4000; // satellite image spans ~4 km
const SKY_R = 1400; // celestial sphere radius
const SUN_R = 70;
const MOON_R = 45;

export interface ReplayFrame {
  date: Date;
  radar: number[][]; // [x lateral starboard, y forward, z, v]
  yawDeg: number; // boat heading (0 = north)
}

export interface ReplayHandle {
  setFrame(f: ReplayFrame): void;
}

function skyColour(elev: number): THREE.Color {
  // night navy -> twilight amber-rose -> day seeblau
  if (elev < -8) return new THREE.Color("#0a1626");
  if (elev < 2) {
    const t = (elev + 8) / 10;
    return new THREE.Color("#0a1626").lerp(new THREE.Color("#e8956b"), t);
  }
  const t = Math.min((elev - 2) / 30, 1);
  return new THREE.Color("#e8956b").lerp(new THREE.Color("#7ec4e8"), t);
}

export function ReplayScene({
  lat,
  lon,
  onReady,
}: {
  lat: number;
  lon: number;
  onReady: (h: ReplayHandle) => void;
}) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0xbfdcec, 1400, SECTOR_M * 1.8);

    const camera = new THREE.PerspectiveCamera(
      55,
      mount.clientWidth / mount.clientHeight,
      0.5,
      SKY_R * 3,
    );
    camera.position.set(34, 9, -12); // NE of the boat, facing the afternoon sun
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 11, 0); // frame boat low, sky high
    controls.maxDistance = 1500;
    controls.minDistance = 4;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;

    // --- ground: satellite sector (top edge of the image = north = -Z)
    const texLoader = new THREE.TextureLoader();
    const satTex = texLoader.load("/replay/satellite_sector.jpg");
    satTex.colorSpace = THREE.SRGBColorSpace;
    // Unlit satellite imagery keeps its true colours; daylight modulates it.
    const groundMat = new THREE.MeshBasicMaterial({ map: satTex });
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(SECTOR_M, SECTOR_M),
      groundMat,
    );
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    // subtle water shimmer overlay over the lake (whole plane, faint)
    const shimmer = new THREE.Mesh(
      new THREE.PlaneGeometry(SECTOR_M, SECTOR_M),
      new THREE.MeshPhysicalMaterial({
        color: 0x2e6f8e,
        transparent: true,
        opacity: 0.04,
        roughness: 0.15,
        metalness: 0.1,
      }),
    );
    shimmer.rotation.x = -Math.PI / 2;
    shimmer.position.y = 0.05;
    scene.add(shimmer);

    // --- lights
    const sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
    scene.add(sunLight);
    const hemi = new THREE.HemisphereLight(0xbdd9ea, 0x3a4a52, 0.7);
    scene.add(hemi);

    // --- sun (NASA model + emissive texture + glow sprite)
    const sunGroup = new THREE.Group();
    scene.add(sunGroup);
    const sunTex = texLoader.load("/replay/sun_texture.jpg");
    sunTex.colorSpace = THREE.SRGBColorSpace;
    new GLTFLoader().load("/replay/sun.glb", (gltf) => {
      const model = gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3()).length() || 1;
      model.scale.setScalar((SUN_R * 2) / size);
      model.traverse((o) => {
        if (o instanceof THREE.Mesh) {
          o.material = new THREE.MeshBasicMaterial({ map: sunTex });
        }
      });
      sunGroup.add(model);
    });
    const glow = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: (() => {
          const c = document.createElement("canvas");
          c.width = c.height = 128;
          const g = c.getContext("2d")!;
          const grad = g.createRadialGradient(64, 64, 8, 64, 64, 64);
          grad.addColorStop(0, "rgba(255,235,180,0.9)");
          grad.addColorStop(0.4, "rgba(255,200,90,0.35)");
          grad.addColorStop(1, "rgba(255,180,60,0)");
          g.fillStyle = grad;
          g.fillRect(0, 0, 128, 128);
          return new THREE.CanvasTexture(c);
        })(),
        transparent: true,
        depthWrite: false,
      }),
    );
    glow.scale.setScalar(SUN_R * 7);
    sunGroup.add(glow);

    // --- moon (NASA LRO topo model, decimated for the web). Lit by the
    // scene's sun light, so its illuminated limb genuinely faces the sun.
    const moon = new THREE.Group();
    scene.add(moon);
    new GLTFLoader().load("/replay/moon.glb", (gltf) => {
      const model = gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3()).length() || 1;
      model.scale.setScalar((MOON_R * 2) / size);
      moon.add(model);
    });

    // --- boat placeholder (hull + mast; swapped for the real model later)
    const boat = new THREE.Group();
    boat.scale.setScalar(4); // presence over strict scale until the real model
    const hull = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.16, 0.9, 6, 12),
      new THREE.MeshStandardMaterial({ color: 0x13314a, roughness: 0.5 }),
    );
    hull.rotation.x = Math.PI / 2;
    hull.position.y = 0.16;
    boat.add(hull);
    const mast = new THREE.Mesh(
      new THREE.CylinderGeometry(0.015, 0.02, 1.6, 8),
      new THREE.MeshStandardMaterial({ color: 0xdddddd }),
    );
    mast.position.y = 1.0;
    boat.add(mast);
    const sail = new THREE.Mesh(
      new THREE.ConeGeometry(0.34, 1.2, 3, 1, true),
      new THREE.MeshStandardMaterial({
        color: 0xf2f2ee,
        side: THREE.DoubleSide,
        roughness: 0.8,
      }),
    );
    sail.position.set(0.12, 1.0, 0);
    sail.rotation.z = 0.06;
    boat.add(sail);
    scene.add(boat);

    // --- sensor FOV fans on the water (fisheye 120°, radar ±55° to 9 m)
    function fov(halfDeg: number, r: number, colour: number, opacity: number) {
      const shape = new THREE.Shape();
      shape.moveTo(0, 0);
      shape.absarc(0, 0, r, Math.PI / 2 - (halfDeg * Math.PI) / 180,
                   Math.PI / 2 + (halfDeg * Math.PI) / 180, false);
      shape.lineTo(0, 0);
      const mesh = new THREE.Mesh(
        new THREE.ShapeGeometry(shape, 48),
        new THREE.MeshBasicMaterial({
          color: colour, transparent: true, opacity, depthWrite: false,
          side: THREE.DoubleSide,
        }),
      );
      mesh.rotation.x = -Math.PI / 2;
      mesh.rotation.z = Math.PI; // shape +Y -> forward (-Z north at yaw 0)
      mesh.position.y = 0.12;
      return mesh;
    }
    // sensor geometry stays TRUE scale (the boat group is scaled for
    // presence), rotating with the boat's yaw via its own group
    const sensorGroup = new THREE.Group();
    scene.add(sensorGroup);
    const fisheyeFov = fov(60, 22, 0x00a9e0, 0.1);
    const radarFov = fov(55, 9, 0x7a6bb5, 0.16);
    sensorGroup.add(fisheyeFov, radarFov);

    // --- radar returns
    const returnsGroup = new THREE.Group();
    sensorGroup.add(returnsGroup);
    const returnGeom = new THREE.SphereGeometry(0.11, 10, 10);
    const returnMat = new THREE.MeshBasicMaterial({ color: 0x1487b8 });

    // --- frame updates
    function setFrame(f: ReplayFrame) {
      const sun = sunPosition(f.date, lat, lon);
      const moonP: HorizontalPos = moonPosition(f.date, lat, lon);
      const sv = horizontalToVector(sun);
      const mv = horizontalToVector(moonP);
      sunGroup.position.set(sv[0] * SKY_R, sv[1] * SKY_R, sv[2] * SKY_R);
      moon.position.set(mv[0] * SKY_R, mv[1] * SKY_R, mv[2] * SKY_R);
      moon.visible = moonP.elevationDeg > -2;
      sunGroup.visible = sun.elevationDeg > -10;
      sunLight.position.copy(sunGroup.position);
      sunLight.intensity = Math.max(0.05, Math.sin((sun.elevationDeg * Math.PI) / 180)) * 2.6;
      hemi.intensity = 0.25 + Math.max(0, Math.sin((sun.elevationDeg * Math.PI) / 180)) * 0.6;
      const sky = skyColour(sun.elevationDeg);
      scene.background = sky;
      const daylight = 0.55 + 0.45 * Math.max(0, Math.sin((sun.elevationDeg * Math.PI) / 180));
      groundMat.color.setScalar(daylight);
      scene.fog!.color = sky.clone().lerp(new THREE.Color(0xffffff), 0.25);

      boat.rotation.y = (-f.yawDeg * Math.PI) / 180;
      sensorGroup.rotation.y = boat.rotation.y;

      returnsGroup.clear();
      for (const p of f.radar) {
        const [x, y] = p;
        if (y == null || y < 0.3) continue;
        const m = new THREE.Mesh(returnGeom, returnMat);
        // boat frame: x starboard, y forward -> local (x, -z)
        m.position.set(x!, 0.35, -y!);
        returnsGroup.add(m);
      }
    }

    onReady({ setFrame });

    let raf = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    const onResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [lat, lon, onReady]);

  return <div ref={mountRef} className="h-full w-full" />;
}
