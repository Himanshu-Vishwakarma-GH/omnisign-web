import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { clone as cloneSkinned } from 'three/examples/jsm/utils/SkeletonUtils.js'
import Avatar from './Avatar'
import {
  boneCurlAngles,
  FINGER_BONES,
  handCurlsFromFrame,
  wristTarget,
  type FingerCurls,
  type FingerName,
  type HandSide,
} from '../lib/handMapping'
import { loadLexicon, restFrame, type Lexicon } from '../lib/lexicon'

interface HandAvatarProps {
  /** Gloss tokens currently being signed, e.g. ["YOUR","NAME","WHAT"]. */
  tokens: string[]
  /** Increment to replay the current tokens. */
  replayKey: number
}

const FRAME_MS = 1000 / 12 // 12 fps playback of lexicon frames
const HAND_SCALE = 0.2 // Model is ~2.6 units long as loaded; scale to ~0.5.
const HAND_URL = `${import.meta.env.BASE_URL}models/hand.glb`

/**
 * Speech-to-Sign avatar: the real rigged "Rigged Hand" GLB model
 * (J-Toastie, CC-BY, Poly Pizza), instantiated twice — the left hand
 * as built, the right hand mirrored (scale.x = -1 on a wrapper group;
 * three.js flips the winding for negative-determinant transforms, so
 * the mirrored hand does not render inside-out).
 *
 * Driven from the SAME 225-dim lexicon frames as the old Avatar:
 * wrist position/orientation from the hand landmarks, finger curl PER
 * FINGER from that finger's tip-to-MCP distance, thumb likewise.
 * Bone rotations are smoothed with SLERP every requestAnimationFrame
 * (~60 FPS) so gestures flow smoothly from 12 fps lexicon frames.
 * If hand.glb fails to load, falls back to the old procedural Avatar.
 */
export default function HandAvatar({ tokens, replayKey }: HandAvatarProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const tokensRef = useRef(tokens)
  tokensRef.current = tokens
  const replayRef = useRef(replayKey)
  replayRef.current = replayKey
  const [loadFailed, setLoadFailed] = useState(false)

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
    camera.position.set(0, 0, 2.4)
    camera.lookAt(0, -0.05, 0)

    scene.add(new THREE.AmbientLight(0xffffff, 0.9))
    const key = new THREE.DirectionalLight(0xffffff, 1.6)
    key.position.set(1.5, 2, 2)
    scene.add(key)
    const rim = new THREE.DirectionalLight(0x22d3ee, 0.8)
    rim.position.set(-2, 1, -1)
    scene.add(rim)

    interface HandRig {
      root: THREE.Group
      bones: Map<string, THREE.Bone>
      bindQuats: Map<string, THREE.Quaternion>
    }

    let rigs: HandRig[] = []
    let cancelled = false

    new GLTFLoader().load(
      HAND_URL,
      (gltf) => {
        if (cancelled) return
        // Wrist (HandMain bone) position in the loaded scene's space,
        // so the hand pivots around its wrist, not the scene origin.
        gltf.scene.updateMatrixWorld(true)
        const handMain = gltf.scene.getObjectByName('HandMain')
        const wristLocal = new THREE.Vector3(1.166, 0.123, 0)
        if (handMain) handMain.getWorldPosition(wristLocal)

        const buildRig = (source: THREE.Object3D, side: HandSide): HandRig => {
          const root = new THREE.Group()
          root.scale.setScalar(HAND_SCALE)
          scene.add(root)
          const mirror = new THREE.Group()
          mirror.scale.x = side === 'right' ? -1 : 1
          root.add(mirror)
          const offset = new THREE.Group()
          offset.position.copy(wristLocal).negate()
          mirror.add(offset)
          offset.add(source)
          const bones = new Map<string, THREE.Bone>()
          const bindQuats = new Map<string, THREE.Quaternion>()
          for (const finger of Object.keys(FINGER_BONES) as FingerName[]) {
            for (const name of FINGER_BONES[finger]) {
              const bone = source.getObjectByName(name) as THREE.Bone | undefined
              if (bone) {
                bones.set(name, bone)
                bindQuats.set(name, bone.quaternion.clone())
              }
            }
          }
          return { root, bones, bindQuats }
        }

        rigs = [
          buildRig(gltf.scene, 'left'),
          buildRig(cloneSkinned(gltf.scene), 'right'),
        ]
      },
      undefined,
      () => {
        if (!cancelled) setLoadFailed(true)
      },
    )

    let lexicon: Lexicon | null = null
    let cancelledLex = false
    void loadLexicon().then((lx) => {
      if (!cancelledLex) lexicon = lx
    })

    let frames: Float32Array[] = [restFrame()]
    let frameIdx = 0
    let lastTokens: string[] = []
    let lastReplay = replayKey
    let lastTick = performance.now()

    const zAxis = new THREE.Vector3(0, 0, 1)
    const tmpQ = new THREE.Quaternion()

    const applyHand = (
      rig: HandRig,
      frame: Float32Array,
      side: HandSide,
      handOffset: number,
    ) => {
      const target = wristTarget(frame, side, handOffset)
      rig.root.position.lerp(target.position, 0.25)
      rig.root.quaternion.slerp(target.quaternion, 0.2)
      const curls: FingerCurls = handCurlsFromFrame(frame, handOffset, side)
      for (const finger of Object.keys(FINGER_BONES) as FingerName[]) {
        const angles = boneCurlAngles(finger, curls[finger])
        FINGER_BONES[finger].forEach((name, i) => {
          const bone = rig.bones.get(name)
          const bind = rig.bindQuats.get(name)
          if (!bone || !bind) return
          tmpQ.setFromAxisAngle(zAxis, angles[i])
          const targetQ = bind.clone().multiply(tmpQ)
          bone.quaternion.slerp(targetQ, 0.2)
        })
      }
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
      if (rigs.length === 2) {
        applyHand(rigs[0], frame, 'left', 33)
        applyHand(rigs[1], frame, 'right', 54)
      }
      // Gentle idle motion so the hands never look frozen.
      scene.position.y = Math.sin(now / 900) * 0.004
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
      cancelledLex = true
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      if (renderer.domElement.parentElement === mount) {
        mount.removeChild(renderer.domElement)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (loadFailed) {
    return <Avatar tokens={tokens} replayKey={replayKey} />
  }
  return <div ref={mountRef} className="h-full w-full" aria-label="Signing hands" />
}
