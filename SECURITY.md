# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it privately:

- **Email:** security@example.com (replace with your email)
- **GitHub:** Use private vulnerability reporting (if enabled)

**Please do NOT:**
- Open public issues for security vulnerabilities
- Disclose the vulnerability publicly until it has been addressed

## What to Include

When reporting a vulnerability, please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

## Response Timeline

- We will acknowledge receipt within 48 hours
- We will provide a detailed response within 7 days
- We will work on a fix and release it as soon as possible

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| Older   | :x:                |

Only the latest version receives security updates.

## Security Best Practices

When using this project:

1. **API Keys:** Never commit API keys or secrets to git
   - Use Modal secrets: `modal secret create wandb WANDB_API_KEY=your_key`
   - Use `.env` files (already in `.gitignore`)

2. **Dependencies:** Keep dependencies updated
   - Run `uv pip list --outdated` regularly
   - Review security advisories for PyTorch, Modal, W&B

3. **Code Execution:** Be cautious with untrusted code
   - Review `modal_app.py` before deployment
   - Audit third-party datasets

4. **Modal Security:**
   - Use Modal secrets for sensitive data
   - Don't log sensitive information
   - Review Modal volume permissions

## Known Limitations

This is an educational project. Do not use in production without:
- Security audit
- Input validation
- Rate limiting
- Access controls
- Monitoring and alerting
