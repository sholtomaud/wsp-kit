import { defineConfig, devices } from '@playwright/test';

// WebGPU on SwiftShader (Chromium's software Vulkan), so the WGSL kernels run
// on a real WebGPU adapter on a headless CI machine without a GPU.
const webgpu = ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'];

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://localhost:5174',
    ...devices['Desktop Chrome'],
    launchOptions: { args: webgpu },
  },
  webServer: {
    command: 'npx vite --port 5174 --strictPort',
    url: 'http://localhost:5174',
    reuseExistingServer: !process.env.CI,
  },
});
