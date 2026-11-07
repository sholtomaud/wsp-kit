import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset URLs, so the build works at https://<user>.github.io/wsp-kit/ and anywhere else.
  base: './',
  build: { target: 'es2022' },
});
