import * as THREE from "three";

/**
 * SuperSplat-style fly camera: WASD move, QE up/down, mouse-look on left-drag,
 * scroll to adjust speed, Shift for 3x speed boost. Pitch clamped to +-89 deg.
 */
export class FlyCamera {
  speed = 2.0;
  readonly keysDown = new Set<string>();
  private isDragging = false;
  private lastMouse: { x: number; y: number } | null = null;
  private euler = new THREE.Euler(0, 0, 0, "YXZ");
  private enabled = false;
  private domElement: HTMLElement | null = null;

  // Bound handlers for clean attach/detach
  private _onKeyDown = this.onKeyDown.bind(this);
  private _onKeyUp = this.onKeyUp.bind(this);
  private _onMouseDown = this.onMouseDown.bind(this);
  private _onMouseMove = this.onMouseMove.bind(this);
  private _onMouseUp = this.onMouseUp.bind(this);
  private _onWheel = this.onWheel.bind(this);
  private _onContextMenu = (e: Event) => e.preventDefault();

  /** Copy camera orientation on mode switch. */
  activate(camera: THREE.PerspectiveCamera): void {
    this.euler.setFromQuaternion(camera.quaternion, "YXZ");
    this.keysDown.clear();
    this.isDragging = false;
    this.lastMouse = null;
    this.enabled = true;
  }

  deactivate(): void {
    this.keysDown.clear();
    this.isDragging = false;
    this.lastMouse = null;
    this.enabled = false;
  }

  /** Attach DOM event listeners. */
  attach(domElement: HTMLElement): void {
    this.domElement = domElement;
    window.addEventListener("keydown", this._onKeyDown);
    window.addEventListener("keyup", this._onKeyUp);
    domElement.addEventListener("mousedown", this._onMouseDown);
    window.addEventListener("mousemove", this._onMouseMove);
    window.addEventListener("mouseup", this._onMouseUp);
    domElement.addEventListener("wheel", this._onWheel, { passive: false });
    domElement.addEventListener("contextmenu", this._onContextMenu);
  }

  /** Detach DOM event listeners. */
  detach(): void {
    window.removeEventListener("keydown", this._onKeyDown);
    window.removeEventListener("keyup", this._onKeyUp);
    if (this.domElement) {
      this.domElement.removeEventListener("mousedown", this._onMouseDown);
      this.domElement.removeEventListener("wheel", this._onWheel);
      this.domElement.removeEventListener("contextmenu", this._onContextMenu);
    }
    window.removeEventListener("mousemove", this._onMouseMove);
    window.removeEventListener("mouseup", this._onMouseUp);
    this.domElement = null;
    this.keysDown.clear();
    this.isDragging = false;
    this.lastMouse = null;
  }

  /** Per-frame update: apply WASD movement + accumulated mouse delta. */
  update(camera: THREE.PerspectiveCamera, dt: number): void {
    if (!this.enabled) return;

    const speedMul = this.keysDown.has("shift") ? 3 : 1;
    const moveSpeed = this.speed * speedMul * dt;

    // Camera-relative directions
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();

    const right = new THREE.Vector3();
    right.crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

    const move = new THREE.Vector3();

    if (this.keysDown.has("w")) move.add(forward.clone().multiplyScalar(moveSpeed));
    if (this.keysDown.has("s")) move.add(forward.clone().multiplyScalar(-moveSpeed));
    if (this.keysDown.has("a")) move.add(right.clone().multiplyScalar(-moveSpeed));
    if (this.keysDown.has("d")) move.add(right.clone().multiplyScalar(moveSpeed));
    if (this.keysDown.has("e")) move.y += moveSpeed;
    if (this.keysDown.has("q")) move.y -= moveSpeed;

    camera.position.add(move);

    // Apply euler to camera quaternion
    camera.quaternion.setFromEuler(this.euler);
  }

  private onKeyDown(e: KeyboardEvent): void {
    if (!this.enabled) return;
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

    const key = e.key.toLowerCase();
    if (key === "shift") {
      this.keysDown.add("shift");
    } else if ("wasdqe".includes(key)) {
      this.keysDown.add(key);
    }
  }

  private onKeyUp(e: KeyboardEvent): void {
    const key = e.key.toLowerCase();
    if (key === "shift") {
      this.keysDown.delete("shift");
    } else if ("wasdqe".includes(key)) {
      this.keysDown.delete(key);
    }
  }

  private onMouseDown(e: MouseEvent): void {
    if (!this.enabled) return;
    if (e.button === 0) {
      this.isDragging = true;
      this.lastMouse = { x: e.clientX, y: e.clientY };
    }
  }

  private onMouseMove(e: MouseEvent): void {
    if (!this.enabled || !this.isDragging || !this.lastMouse) return;

    const dx = e.clientX - this.lastMouse.x;
    const dy = e.clientY - this.lastMouse.y;
    this.lastMouse = { x: e.clientX, y: e.clientY };

    const sensitivity = 0.002;
    this.euler.y -= dx * sensitivity;
    this.euler.x -= dy * sensitivity;

    // Clamp pitch to +-89 degrees
    const limit = (89 * Math.PI) / 180;
    this.euler.x = Math.max(-limit, Math.min(limit, this.euler.x));
  }

  private onMouseUp(e: MouseEvent): void {
    if (e.button === 0) {
      this.isDragging = false;
      this.lastMouse = null;
    }
  }

  private onWheel(e: WheelEvent): void {
    if (!this.enabled) return;
    e.preventDefault();
    // Scroll adjusts speed: up = faster, down = slower
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    this.speed = Math.max(0.1, Math.min(50, this.speed * factor));
  }
}
