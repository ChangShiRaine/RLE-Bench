"""Release GL resources in their owning context before a hard reset."""

from robosuite.utils.binding_utils import MjRenderContextOffscreen
from OpenGL import EGL


class PocketRenderContext(MjRenderContextOffscreen):
    def __init__(self, *args, **kwargs):
        self._closed = False
        super().__init__(*args, **kwargs)
        self._release_thread()

    @staticmethod
    def _release_thread():
        # Service warm-up and successive client connections use different threads.
        EGL.eglMakeCurrent(EGL.eglGetCurrentDisplay(), EGL.EGL_NO_SURFACE,
                          EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)

    def render(self, *args, **kwargs):
        try:
            return super().render(*args, **kwargs)
        except Exception:
            self._release_thread()
            raise

    def read_pixels(self, *args, **kwargs):
        self.gl_ctx.make_current()
        try:
            return super().read_pixels(*args, **kwargs)
        finally:
            self._release_thread()

    def upload_texture(self, *args, **kwargs):
        try:
            return super().upload_texture(*args, **kwargs)
        finally:
            self._release_thread()

    def close(self):
        if self._closed:
            return
        self.gl_ctx.make_current()
        self.con.free()
        self.gl_ctx.free()
        self._closed = True

    def __del__(self):
        # Normal teardown is explicit; partially constructed contexts may lack con.
        if hasattr(self, "con"):
            self.close()
