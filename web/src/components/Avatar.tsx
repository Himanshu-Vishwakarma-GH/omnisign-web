import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { loadLexicon, restFrame, type Lexicon } from '../lib/lexicon'

interface AvatarProps {
  /** Gloss tokens currently being signed, e.g. ["YOUR","NAME","WHAT"]. */
  tokens: string[]
  /** Increment to replay the current tokens. */
  replayKey: number
}

const FRAME_MS = 1000 / 12 // 12 fps playback of lexicon frames

/**
 * Procedural Three.js upper-body avatar, RETARGETED from the same 225-dim
 * lexicon frames as the 2D skeleton view (Pipeline B, layer 2).
 *
 * Retargeting: shoulder/elbow quaternions are derived from the pose
 * landmarks (shoulder -> elbow -> wrist vectors); hand openness from the
 * finger-tip-to-wrist distance in the hand landmarks; head yaw/pitch from
 * the nose/ear landmarks. Poses are smoothed with quaternion SLERP at
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

    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.26, 0.42, 8, 16), shirt)
    torso.position.y = -0.28
    root.add(torso)
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 32, 32), skin)
    head.position.y = 0.42
    root.add(head)
    const hair = new THREE.Mesh(
      new THREE.SphereGeometry(0.175, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.55),
      dark,
    )
    hair.position.y = 0.45
    root.add(hair)
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

    let lexicon: Lexicon | null = null
    let cancelled = false
    void loadLexicon().then((lx) => {
      if (!cancelled) lexicon = lx
    })

    let frames: Float32Array[] = [restFrame()]
    let frameIdx = 0
    let lastTokens: string[] = []
    let lastReplay = replayKey
    let lastTick = performance.now()

    const v = (frame: Float32Array, landmark: number): THREE.Vector3 =>
      new THREE.Vector3(
        (frame[landmark * 3] - 0.5) * 2.2,
        (0.5 - frame[landmark * 3 + 1]) * 2.2,
        frame[landmark * 3 + 2] * 2,
      )

    /** Hand openness 0 (fist) .. 1 (open) from fingertip spread. */
    const handOpenness = (frame: Float32Array, handOffset: number): number => {
      const wrist = v(frame, handOffset)
      let spread = 0
      for (const tip of [8, 12, 16, 20]) {
        spread += v(frame, handOffset + tip).distanceTo(wrist)
      }
      // Rest-open spread is ~0.55 in avatar units; fist is ~0.25.
      return Math.min(1, Math.max(0, (spread - 0.25) / 0.3))
    }

    /** Target shoulder/elbow quaternions for one arm from pose landmarks. */
    function armTargets(
      frame: Float32Array,
      shoulderIdx: number,
      elbowIdx: number,
      wristIdx: number,
    ): { shoulder: THREE.Quaternion; elbow: THREE.Quaternion } {
      const shoulder = v(frame, shoulderIdx)
      const elbow = v(frame, elbowIdx)
      const wrist = v(frame, wristIdx)
      const upper = elbow.clone().sub(shoulder).normalize()
      const fore = wrist.clone().sub(elbow).normalize()
      const down = new THREE.Vector3(0, -1, 0)
      const shoulderQ = new THREE.Quaternion().setFromUnitVectors(down, upper)
      const elbowQ = new THREE.Quaternion().setFromUnitVectors(upper, fore)
      return { shoulder: shoulderQ, elbow: elbowQ }
    }

    let raf = 0
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      if (tokensRef.current !== lastTokens || replayRef.current !== lastReplay) {
        lastTokens = tokensRef.current
        lastReplay = replayRef.current
        if (lexicon && tokensRef.current.length > 0) {
          frames = lexicon.framesForTokens(tokensRef.current).frames
        } else {
          frames = [restFrame()]
        }
        frameIdx = 0
        lastTick = now
      }
      if (now - lastTick >= FRAME_MS && frames.length > 1) {
        frameIdx = (frameIdx + 1) % frames.length
        lastTick = now
      }
      const frame = frames[Math.min(frameIdx, frames.length - 1)]
      // Retarget both arms.
      const left = armTargets(frame, 11, 13, 15)
      const right = armTargets(frame, 12, 14, 16)
      leftArm.shoulder.quaternion.slerp(left.shoulder, 0.25)
      leftArm.elbow.quaternion.slerp(left.elbow, 0.25)
      rightArm.shoulder.quaternion.slerp(right.shoulder, 0.25)
      rightArm.elbow.quaternion.slerp(right.elbow, 0.25)
      // Fingers: curl from openness (hand offsets 33 = left, 54 = right).
      const openL = handOpenness(frame, 33)
      const openR = handOpenness(frame, 54)
      const curlArm = (rig: ArmRig, open: number) => {
        rig.fingers.forEach((f, i) => {
          f.rotation.x = (1 - open) * (i === 4 ? 0.5 : 1.1)
        })
      }
      curlArm(leftArm, openL)
      curlArm(rightArm, openR)
      // Head: yaw from nose/ear x, pitch from nose y.
      const noseX = frame[0]
      const noseY = frame[1]
      const headQ = new THREE.Quaternion().setFromEuler(
        new THREE.Euler((noseY - 0.155) * 2, (noseX - 0.5) * -2.5, 0),
      )
      head.quaternion.slerp(headQ, 0.12)
      // Gentle idle breathing so the avatar never looks frozen.
      root.position.y = Math.sin(now / 900) * 0.008
      renderer.render(scene, camera)
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
      cancelled = true
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return <div ref={mountRef} className="h-full w-full" aria-label="Signing avatar" />
}
