// public/elements/ThreeDmolViewer.jsx (updated)
import { useEffect, useRef } from 'react';

export default function ThreeDmolViewer() {
  const containerRef = useRef(null);
  const {
    pdbContent = '',
    cifContent = '',
    width = '600px',
    height = '400px',
    backgroundColor = 'black',
  } = props || {};

  useEffect(() => {
    console.log('Received props:', { pdbContentLength: pdbContent.length, cifContentLength: cifContent.length });

    // Multi-CDN fallback: try reliable CDNs first, Pitt server as last resort
    const CDN_SOURCES = [
      'https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js',   // jsDelivr (Cloudflare)
      'https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.5.3/3Dmol-min.js', // cdnjs (Cloudflare)
      'https://3dmol.csb.pitt.edu/build/3Dmol-min.js',                  // Original Pitt (fallback)
    ];

    function loadScriptWithFallback(sources, index = 0) {
      if (index >= sources.length) {
        console.error('Failed to load 3Dmol.js from all CDN sources');
        return;
      }
      const script = document.createElement('script');
      script.src = sources[index];
      script.async = true;
      script.onload = () => {
        console.log(`3Dmol.js loaded from: ${sources[index]}`);
        initializeViewer();
      };
      script.onerror = () => {
        console.warn(`Failed to load 3Dmol.js from: ${sources[index]}, trying next source...`);
        document.head.removeChild(script);
        loadScriptWithFallback(sources, index + 1);
      };
      document.head.appendChild(script);
    }

    if (!window.$3Dmol) {
      loadScriptWithFallback(CDN_SOURCES);
    } else {
      initializeViewer();
    }

    function initializeViewer() {
      if (!containerRef.current || !window.$3Dmol) return;

      try {
        const viewer = window.$3Dmol.createViewer(containerRef.current, {
          backgroundColor: backgroundColor,
        });

        let content = '';
        let format = '';

        if (pdbContent && pdbContent.trim()) {
          content = pdbContent;
          format = 'pdb';
          console.log('Using PDB content');
        } else if (cifContent && cifContent.trim()) {
          content = cifContent;
          format = 'cif';  // ← 3Dmol.js supports 'cif' format
          console.log('Using CIF content');
        } else {
          console.warn('No valid content (PDB or CIF) provided');
          return;
        }

        const model = viewer.addModel(content, format);
        if (!model) {
          console.error(`Failed to add model - invalid ${format} data`);
          return;
        }

        viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
        viewer.zoomTo();
        viewer.render();
      } catch (error) {
        console.error('3Dmol viewer error:', error);
      }
    }

    return () => {
      // Cleanup
      if (window.$3Dmol && containerRef.current) {
        const viewer = window.$3Dmol.getViewer(containerRef.current);
        if (viewer) viewer.removeAllModels();
      }
    };
  }, [pdbContent, cifContent, backgroundColor]);

  return (
    <div
      ref={containerRef}
      style={{
        width,
        height,
        position: 'relative',
        border: '1px solid #ddd',
      }}
    />
  );
}
