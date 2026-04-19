// Sidebar TOC scrollspy + mobile toggle
(function () {
  const toc = document.querySelector('aside.toc');
  const toggle = document.querySelector('.toc-toggle');
  const links = Array.from(document.querySelectorAll('aside.toc a[href^="#"]'));
  const sections = links
    .map(a => document.getElementById(a.getAttribute('href').slice(1)))
    .filter(Boolean);

  // Mobile toggle
  if (toggle && toc) {
    toggle.addEventListener('click', () => toc.classList.toggle('open'));
    // Close on link tap (mobile)
    links.forEach(a => a.addEventListener('click', () => toc.classList.remove('open')));
  }

  // Scrollspy — mark the TOC link whose section is topmost in view
  const setActive = () => {
    const y = window.scrollY + 120; // offset for sticky header
    let current = sections[0];
    for (const s of sections) {
      if (s.offsetTop <= y) current = s;
    }
    links.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === '#' + current.id);
    });
  };

  let rafId = null;
  window.addEventListener('scroll', () => {
    if (rafId) return;
    rafId = requestAnimationFrame(() => {
      setActive();
      rafId = null;
    });
  });
  setActive();
})();
