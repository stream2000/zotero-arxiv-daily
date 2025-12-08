from paper import BasePaper
import math
from tqdm import tqdm
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib
import datetime
import time
from loguru import logger
import concurrent.futures # Added for concurrency

framework = """
<!DOCTYPE HTML>
<html>
<head>
  <meta charset="utf-8">
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(title:str, authors:str, rate:str,arxiv_id:str, abstract:str, pdf_url:str, code_url:str=None, affiliations:str=None):
    code = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>' if code_url else ''
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">{arxiv_id}</a>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {abstract}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
            {code}
        </td>
    </tr>
</table>
"""
    return block_template.format(title=title, authors=authors,rate=rate,arxiv_id=arxiv_id, abstract=abstract, pdf_url=pdf_url, code=code, affiliations=affiliations)

def get_stars(score:float):
    if score is None:
        return '' # Return empty string if score is None
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'

def _render_single_paper_block(p: BasePaper, db):
    """Helper function to render a single paper block, including TLDR generation."""
    # Check if TLDR needs to be generated and saved
    should_save = not p.has_tldr
    
    # Accessing p.tldr will generate it if not cached
    tldr_text = p.tldr 
    
    if should_save:
        # Save to DB
        # Note: In high concurrency, this might lock the DB file temporarily, 
        # but sqlite3 handles this with timeouts usually.
        try:
            db.update_tldr(p.arxiv_id, tldr_text)
        except Exception as e:
            logger.error(f"Failed to save TLDR for {p.arxiv_id}: {e}")

    rate = get_stars(p.score)
    author_list = [a.name for a in p.authors]
    num_authors = len(author_list)
    
    if num_authors <= 5:
        authors = ', '.join(author_list)
    else:
        authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
    
    affiliations_str = 'Unknown Affiliation'
    if p.affiliations is not None:
        affiliations_str = p.affiliations[:5]
        affiliations_str = ', '.join(affiliations_str)
        if len(p.affiliations) > 5:
            affiliations_str += ', ...'
            
    return get_block_html(p.title, authors, rate, p.arxiv_id, tldr_text, p.pdf_url, p.code_url, affiliations_str)

def generate_report(papers:list[BasePaper], db):
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())
    
    # Store results in a list initialized with None to maintain order
    results = [None] * len(papers)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Submit all tasks and keep track of their index in the original list
        future_to_index = {executor.submit(_render_single_paper_block, p, db): i for i, p in enumerate(papers)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(papers), desc='Generating Report (concurrently)'):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                logger.error(f'Paper block generation generated an exception: {exc}')
                results[index] = get_block_html("Error rendering paper", "N/A", "", "N/A", f"Error: {exc}", "#", "#", "")

    content = '<br>' + '</br><br>'.join(results) + '</br>'
    return framework.replace('__CONTENT__', content)

def send_email(sender:str, receiver:str, password:str,smtp_server:str,smtp_port:int, html:str,):
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.warning(f"Failed to use TLS. {e}")
        logger.warning(f"Try to use SSL.")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
