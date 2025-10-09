/**
 * Details Toggle Functionality
 * Provides toggle functionality for <details> tags in Jekyll posts
 * Works independently of theme CSS and JavaScript
 */

(function() {
    'use strict';

    // Wait for DOM to be ready
    function ready(fn) {
        if (document.readyState !== 'loading') {
            fn();
        } else {
            document.addEventListener('DOMContentLoaded', fn);
        }
    }

    // Initialize details toggle functionality
    function initDetailsToggle() {
        const detailsElements = document.querySelectorAll('details');
        
        if (detailsElements.length === 0) {
            return; // No details elements found
        }

        detailsElements.forEach(function(details) {
            const summary = details.querySelector('summary');
            
            if (!summary) {
                return; // No summary found
            }

            // Add click event listener to summary
            summary.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Toggle the details element
                if (details.hasAttribute('open')) {
                    details.removeAttribute('open');
                } else {
                    details.setAttribute('open', '');
                }
            });

            // Add visual indicator
            if (!details.hasAttribute('open')) {
                details.removeAttribute('open');
            }
        });
    }

    // Add CSS styles for better visual feedback
    function addToggleStyles() {
        const style = document.createElement('style');
        style.textContent = `
            details {
                margin: 1em 0;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                padding: 0;
            }
            
            details summary {
                padding: 8px 16px;
                background-color: #f6f8fa;
                border-bottom: 1px solid #e1e4e8;
                cursor: pointer;
                font-weight: 600;
                list-style: none;
                position: relative;
                user-select: none;
            }
            
            details summary::-webkit-details-marker {
                display: none;
            }
            
            details summary::before {
                content: "▶";
                margin-right: 8px;
                transition: transform 0.2s ease;
                display: inline-block;
            }
            
            details[open] summary::before {
                transform: rotate(90deg);
            }
            
            details[open] summary {
                border-bottom: 1px solid #e1e4e8;
            }
            
            details > *:not(summary) {
                padding: 16px;
                margin: 0;
            }
            
            details[open] > *:not(summary) {
                display: block;
            }
            
            details:not([open]) > *:not(summary) {
                display: none;
            }
            
            /* Hover effects */
            details summary:hover {
                background-color: #f1f3f4;
            }
            
            /* Focus styles for accessibility */
            details summary:focus {
                outline: 2px solid #0366d6;
                outline-offset: -2px;
            }
        `;
        
        document.head.appendChild(style);
    }

    // Initialize everything when DOM is ready
    ready(function() {
        addToggleStyles();
        initDetailsToggle();
    });

})();
