document.addEventListener('DOMContentLoaded', () => {
  const links = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('.doc-section');
  const menuToggle = document.getElementById('menuToggle');
  const sidebar = document.getElementById('sidebar');

  // ==========================================
  // 1. Theme Management (Light / Dark)
  // ==========================================
  const themeToggleBtns = document.querySelectorAll('.theme-toggle-btn');
  const savedTheme = localStorage.getItem('nio-theme');
  const moonIconSvg = '<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M6 .278a.768.768 0 0 1 .08.858 7.208 7.208 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277.527 0 1.04-.055 1.533-.16a.787.787 0 0 1 .81.316.733.733 0 0 1-.031.893A8.349 8.349 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.752.752 0 0 1 6 .278z"/></svg>';
  const sunIconSvg = '<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM8 0a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0V.75A.75.75 0 0 1 8 0zm0 13a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 8 13zM2.343 2.343a.75.75 0 0 1 1.061 0l1.06 1.061a.75.75 0 0 1-1.06 1.06l-1.06-1.06a.75.75 0 0 1 0-1.061zm9.193 9.193a.75.75 0 0 1 1.06 0l1.061 1.06a.75.75 0 0 1-1.06 1.061l-1.061-1.06a.75.75 0 0 1 0-1.061zM0 8a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5H.75A.75.75 0 0 1 0 8zm13 0a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5h-1.5A.75.75 0 0 1 13 8zM2.343 13.657a.75.75 0 0 1 0-1.06l1.06-1.061a.75.75 0 1 1 1.061 1.06l-1.06 1.061a.75.75 0 0 1-1.061 0zm9.193-9.193a.75.75 0 0 1 0-1.06l1.061-1.061a.75.75 0 0 1 1.06 1.06l-1.06 1.061a.75.75 0 0 1-1.061 0z"/></svg>';

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('nio-theme', theme);
    themeToggleBtns.forEach(btn => {
      const textSpan = btn.querySelector('.theme-text');
      const iconSpan = btn.querySelector('.theme-icon');
      if (textSpan) {
        textSpan.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
      }
      if (iconSpan) {
        iconSpan.innerHTML = theme === 'dark' ? sunIconSvg : moonIconSvg;
      }
    });
  }

  // Initialize theme: Default to light mode unless explicitly saved as dark
  if (savedTheme === 'dark') {
    setTheme('dark');
  } else {
    setTheme('light');
  }

  themeToggleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      setTheme(current === 'dark' ? 'light' : 'dark');
    });
  });

  // ==========================================
  // 2. Mobile Menu Toggle
  // ==========================================
  if (menuToggle && sidebar) {
    menuToggle.addEventListener('click', () => {
      const open = sidebar.classList.toggle('open');
      menuToggle.setAttribute('aria-expanded', String(open));
    });

    links.forEach(link => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 900) {
          sidebar.classList.remove('open');
          menuToggle.setAttribute('aria-expanded', 'false');
        }
      });
    });
  }

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && sidebar?.classList.contains('open')) {
      sidebar.classList.remove('open');
      menuToggle?.setAttribute('aria-expanded', 'false');
      menuToggle?.focus();
    }
  });

  // ==========================================
  // 3. Tab Groups (Agent Switcher, etc.)
  // ==========================================
  document.querySelectorAll('.tabs-container').forEach(container => {
    const tabBtns = container.querySelectorAll('.tab-btn');
    const tabPanels = container.querySelectorAll('.tab-panel');

    tabBtns.forEach((btn, index) => {
      btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanels.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        if (tabPanels[index]) {
          tabPanels[index].classList.add('active');
        }
      });
    });
  });

  // ==========================================
  // 4. Copy Code to Clipboard
  // ==========================================
  document.querySelectorAll('.code-wrapper').forEach(wrapper => {
    const copyBtn = wrapper.querySelector('.copy-btn');
    const codeEl = wrapper.querySelector('pre code') || wrapper.querySelector('pre');

    if (copyBtn && codeEl) {
      copyBtn.addEventListener('click', async () => {
        const text = codeEl.innerText.trim();
        try {
          await navigator.clipboard.writeText(text);
          const originalHtml = copyBtn.innerHTML;
          copyBtn.classList.add('copied');
          copyBtn.innerHTML = '✓ Copied';
          setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.innerHTML = originalHtml;
          }, 2000);
        } catch (err) {
          console.error('Failed to copy', err);
        }
      });
    }
  });

  // ==========================================
  // 5. Active Link Scroll Spy
  // ==========================================
  const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -70% 0px',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        links.forEach(link => {
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('active');
          } else {
            link.classList.remove('active');
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => observer.observe(section));
});
