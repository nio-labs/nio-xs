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
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('nio-theme', theme);
    themeToggleBtns.forEach(btn => {
      const textSpan = btn.querySelector('.theme-text');
      if (textSpan) {
        textSpan.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
      }
    });
  }

  // Initialize theme
  if (savedTheme) {
    setTheme(savedTheme);
  } else if (prefersDark.matches) {
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
