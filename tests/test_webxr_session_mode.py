import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp.test_utils import make_mocked_request
from aiohttp.web import HTTPFound
from vuer.server import Vuer


ROOT = Path(__file__).resolve().parents[1]
TELEVUER = ROOT / 'teleop/televuer/src/televuer/televuer.py'
CLIENT = ROOT.parent / '.venv-xr/lib/python3.10/site-packages/vuer/client_build'
BUILD = CLIENT / 'assets/xr-session-fix'
tree = ast.parse(TELEVUER.read_text())
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'webxr_session_mode_for_display')
method = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'TeleVuer' for n in c.body if isinstance(n, ast.FunctionDef) and n.name == '_install_webxr_mode_index')
namespace = {'Path': Path}
exec(compile(ast.Module(body=[helper, method], type_ignores=[]), str(TELEVUER), 'exec'), namespace)


class WebXRSessionModeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.vuer = SimpleNamespace(client_root=CLIENT)
        self.tele = SimpleNamespace(vuer=self.vuer, webxr_session_mode='immersive-ar')
        namespace['_install_webxr_mode_index'](self.tele)

    def test_display_modes(self):
        mode = namespace['webxr_session_mode_for_display']
        self.assertEqual(mode('pass-through'), 'immersive-ar')
        self.assertEqual(mode('ego'), 'immersive-ar')
        self.assertEqual(mode('immersive'), 'immersive-vr')

    async def test_redirect_preserves_websocket_address(self):
        request = make_mocked_request('GET', '/?ws=wss://example.test:8012&xrMode=immersive-vr')
        with self.assertRaises(HTTPFound) as caught:
            await self.vuer.socket_index(request)
        self.assertIn('ws=wss://example.test:8012', caught.exception.location)
        self.assertIn('xrMode=immersive-ar', caught.exception.location)
        self.assertNotIn('immersive-vr', caught.exception.location)

    async def test_page_uses_one_build_and_visible_loading_status(self):
        response = await self.vuer.socket_index(make_mocked_request('GET', '/?xrMode=immersive-ar'))
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertIn('window.__TELEVUER_XR_MODE__="immersive-ar"', response.text)
        self.assertIn('正在加载遥操页面', response.text)
        urls = re.findall(r'(?:href|src)="([^"]+)"', response.text)
        self.assertTrue(urls)
        for url in urls:
            self.assertTrue(url.startswith('/assets/xr-session-fix/'), url)
            self.assertTrue((CLIENT / url.lstrip('/')).is_file(), url)

    async def test_websocket_upgrade_bypasses_html_redirect(self):
        request = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket'})
        with patch.object(Vuer, 'socket_index', new_callable=AsyncMock) as handler:
            handler.return_value = 'websocket-response'
            self.assertEqual(await self.vuer.socket_index(request), 'websocket-response')
            handler.assert_awaited_once_with(self.vuer, request)

    def test_all_module_imports_share_one_core(self):
        core = BUILD / 'chunks/chunk-Dd3xtWba.js'
        self.assertTrue(core.is_file())
        references = 0
        for file in BUILD.rglob('*.js'):
            source = file.read_text()
            refs = re.findall(r'''(?:from\s*|import\s*\(?)["']([^"']+)["']''', source)
            for ref in refs:
                if not ref.startswith('.'):
                    continue
                target = (file.parent / ref).resolve()
                self.assertTrue(target.is_file(), (file, ref))
                if 'chunk-Dd3xtWba' in ref:
                    references += 1
                    self.assertEqual(target, core.resolve())
            self.assertNotRegex(source, r'https?://[^"\s]*drei-assets/xr-session-fix/')
        self.assertGreater(references, 1)


if __name__ == '__main__':
    unittest.main()
