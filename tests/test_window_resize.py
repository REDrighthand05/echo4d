"""Exercise the shipped desktop page in Edge, with no desktop bridge required."""
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    url = (Path(__file__).resolve().parents[1] / "web_ui/index.html").as_uri()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            **({"executable_path": os.environ["ECHO_TEST_CHROMIUM"]}
               if "ECHO_TEST_CHROMIUM" in os.environ else {"channel": "msedge"}), args=[
            "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
        ])
        page = browser.new_page(viewport={"width": 900, "height": 600})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("typeof gpuOK !== 'undefined' && gpuOK && !!gpu.fbos")
        page.evaluate("""() => {
          window.originalDocument = document;
          window.originalResources = [...gpu.texState, ...gpu.fbos, ...gpu.texStatics];
          window.resizeCalls = 0;
          const original = rescaleField;
          rescaleField = () => { resizeCalls++; original(); };
        }""")
        for width, height in [(1200, 800), (900, 600)] * 5:
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_function("([w,h]) => W === w && H === h", arg=[width, height])
        page.evaluate("""() => {
          window.beforeBurst = resizeCalls;
          for (let i = 0; i < 100; i++) window.dispatchEvent(new Event('resize'));
        }""")
        page.wait_for_function("resizeCalls > beforeBurst")
        assert page.evaluate("resizeCalls === beforeBurst + 1")
        assert page.evaluate("document === originalDocument")
        assert page.evaluate("""originalResources.every((resource, index) =>
          resource === [...gpu.texState, ...gpu.fbos, ...gpu.texStatics][index])""")
        assert page.evaluate("gl.getError() === gl.NO_ERROR")
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main()
