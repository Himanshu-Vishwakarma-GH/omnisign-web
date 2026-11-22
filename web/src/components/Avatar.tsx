import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { defaultPoses, type AvatarPose } from '../lib/poses'

interface AvatarProps {
  /** Gloss tokens currently being signed, e.g. ["YOUR","NAME","WHAT"]. */
  tokens: string[]
  /** Increment to replay the current tokens. */
  replayKey: number
}

const POSE_HOLD_MS = 900

/**
 * Procedural Three.js upper-body avatar.
 * No external assets: head, torso, arms and hands are built from spheres,
 * capsules and boxes. Poses are interpolated with quaternion SLERP at
 * requestAnimationFrame speed (~60 FPS).
 */
export default function Avatar({ tokens, replayKey }: AvatarProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const tokensRef = useRef(tokens)
  tokensRef.current = tokens
  const replayRef = useRef(replayKey)
  replayRef.current = replayKey

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    mount.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(
      38,
      mount.clientWidth / mount.clientHeight,
      0.1,
      100,
    )
    camera.position.set(0, 0.35, 2.6)
    camera.lookAt(0, 0.15, 0)

    scene.add(new THREE.AmbientLight(0xffffff, 0.9))
    const key = new THREE.DirectionalLight(0xffffff, 1.6)
    key.position.set(1.5, 2, 2)
    scene.add(key)
    const rim = new THREE.DirectionalLight(0x22d3ee, 0.8)
    rim.position.set(-2, 1, -1)
    scene.add(rim)

    const skin = new THREE.MeshStandardMaterial({ color: 0xe8b88a, roughness: 0.6 })
    const shirt = new THREE.MeshStandardMaterial({ color: 0x155e75, roughness: 0.8 })
    const dark = new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.7 })

    const root = new THREE.Group()
    scene.add(root)

    // Torso
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.26, 0.42, 8, 16), shirt)
    torso.position.y = -0.28
    root.add(torso)
    // Head
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 32, 32), skin)
    head.position.y = 0.42
    root.add(head)
    const hair = new THREE.Mesh(new THREE.SphereGeometry(0.175, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.55), dark)
    hair.position.y = 0.45
    root.add(hair)
    // Eyes
    for (const sx of [-0.06, 0.06]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 12), dark)
      eye.position.set(sx, 0.44, 0.155)
      root.add(eye)
    }

    interface ArmRig {
      shoulder: THREE.Group
      elbow: THREE.Group
      hand: THREE.Group
      fingers: THREE.Mesh[]
    }
    function buildArm(side: 1 | -1): ArmRig {
      const shoulder = new THREE.Group()
      shoulder.position.set(0.3 * side, 0.02, 0)
      root.add(shoulder)
      const upper = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.26, 6, 12), shirt)
      upper.position.y = -0.17
      shoulder.add(upper)
      const elbow = new THREE.Group()
      elbow.position.y = -0.34
      shoulder.add(elbow)
      const fore = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.24, 6, 12), skin)
      fore.position.y = -0.15
      elbow.add(fore)
      const hand = new THREE.Group()
      hand.position.y = -0.32
      elbow.add(hand)
      const palm = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.14, 0.04), skin)
      hand.add(palm)
      const fingers: THREE.Mesh[] = []
      for (let f = 0; f < 4; f++) {
        const finger = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.09, 0.02), skin)
        finger.position.set(-0.04 + f * 0.027, -0.11, 0)
        hand.add(finger)
        fingers.push(finger)
      }
      const thumb = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.07, 0.02), skin)
      thumb.position.set(0.07 * side, -0.03, 0.01)
      thumb.rotation.z = -0.6 * side
      hand.add(thumb)
      fingers.push(thumb)
      return { shoulder, elbow, hand, fingers }
    }

    const leftArm = buildArm(-1)
    const rightArm = buildArm(1)

    // --- pose state -------------------------------------------------------
    let poses: Record<string, AvatarPose> = defaultPoses()
    fetch(`${import.meta.env.BASE_URL}poses.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (j) poses = j
      })
      .catch(() => undefined) // fallback poses already in place

    const neutral: AvatarPose = {
      left: { shoulder: [0.1, 0, 0.15], elbow: 0.35, hand: 0.6 },
      right: { shoulder: [0.1, 0, -0.15], elbow: 0.35, hand: 0.6 },
      head: [0, 0, 0],
    }
    let current: AvatarPose = JSON.parse(JSON.stringify(neutral))
    let target: AvatarPose = neutral
    let tokenQueue: string[] = []
    let poseStart = 0
    let lastReplay = replayKey
    let lastTokens: string[] = []

    const slerpQuat = (a: THREE.Quaternion, b: THREE.Quaternion, t: number) =>
      a.clone().slerp(b, t)

    function applyPose(pose: AvatarPose) {
      const setArm = (rig: ArmRig, p: AvatarPose['left'], mirror: boolean) => {
        const q = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(p.shoulder[0], p.shoulder[1], mirror ? -p.shoulder[2] : p.shoulder[2]),
        )
        rig.shoulder.quaternion.copy(slerpQuat(rig.shoulder.quaternion, q, 0.18))
        const eq = new THREE.Quaternion().setFromEuler(new THREE.Euler(-p.elbow, 0, 0))
        rig.elbow.quaternion.copy(slerpQuat(rig.elbow.quaternion, eq, 0.18))
        const open = p.hand
        rig.fingers.forEach((f, i) => {
          f.rotation.x = (1 - open) * (i === 4 ? 0.5 : 1.1)
        })
      }
      setArm(leftArm, pose.left, true)
      setArm(rightArm, pose.right, false)
      const hq = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(pose.head[0], pose.head[1], pose.head[2]),
      )
      head.quaternion.copy(slerpQuat(head.quaternion, hq, 0.12))
    }

    function poseFor(token: string): AvatarPose {
      if (poses[token]) return poses[token]
      // Unknown word -> finger-spell it letter by letter.
      return poses[token[0]] ?? neutral
    }

    function expandTokens(list: string[]): string[] {
      const out: string[] = []
      for (const t of list) {
        if (poses[t]) out.push(t)
        else for (const ch of t) out.push(ch)
      }
      return out
    }

    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      // Pick up a new token list (or replay) when it changes.
      if (tokensRef.current !== lastTokens || replayRef.current !== lastReplay) {
        lastReplay = replayRef.current
        lastTokens = tokensRef.current
        tokenQueue = expandTokens(tokensRef.current)
        poseStart = now
        target = tokenQueue.length > 0 ? poseFor(tokenQueue[0]) : neutral
      }
      const elapsed = now - poseStart
      if (tokenQueue.length > 0 && elapsed > POSE_HOLD_MS) {
        tokenQueue = tokenQueue.slice(1)
        poseStart = now
        target = tokenQueue.length > 0 ? poseFor(tokenQueue[0]) : neutral
      }
      applyPose(target)
      // Gentle idle breathing so the avatar never looks frozen.
      root.position.y = Math.sin(now / 900) * 0.008
      renderer.render(scene, camera)
      last = now
    }
    raf = requestAnimationFrame(tick)

    const onResize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return <div ref={mountRef} className="h-full w-full" aria-label="Signing avatar" />
}
