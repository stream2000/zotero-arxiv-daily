from paper import BasePaper
import math
from tqdm import tqdm
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

def get_block_html(title:str, authors:str, rate:str,arxiv_id:str, tldr_abstract:str, pdf_url:str, code_url:str=None, affiliations:str=None, citation_count:int=None, published_date:str=None, categories:list=None, summary:str=None, title_zh:str=None):
    code = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px; margin-left: 8px;">Code</a>' if code_url else ''
    citation = f'<strong>Citations:</strong> {citation_count}' if citation_count is not None else ''
    published = f'<strong>Published:</strong> {published_date}' if published_date else ''
    categories_str = f'<strong>Categories:</strong> {", ".join(categories)}' if categories else ''
    abstract_details = f"""
    <details>
        <summary style="cursor: pointer; font-weight: bold;">Show Abstract</summary>
        <p style="text-align: justify;">{summary}</p>
    </details>
    """ if summary else ''
    
    title_block = f"""
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        {title}<br>
        <span style="font-size: 16px; font-weight: normal;">{title_zh}</span>
    </td>
    """ if title_zh else f"""
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        {title}
    </td>
    """

    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        {title_block}
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
            <strong>Relevance:</strong> {rate} &nbsp;&nbsp; {citation}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            {published}
            <br>
            {categories_str}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>arXiv ID:</strong> <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank">{arxiv_id}</a>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr_abstract}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            {abstract_details}
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
    return block_template.format(
        title_block=title_block,
        title=title,
        authors=authors,
        rate=rate,
        arxiv_id=arxiv_id,
        tldr_abstract=tldr_abstract,
        pdf_url=pdf_url,
        code=code,
        affiliations=affiliations,
        citation=citation,
        published=published,
        categories_str=categories_str,
        abstract_details=abstract_details
    )

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
    # This will trigger the LLM call and cache the JSON string if not already done.
    tldr_json_str = p.tldr 
    
    # Save to DB if it was newly generated
    if not p.has_tldr:
        try:
            # We save the raw JSON string that p.tldr returns
            db.update_tldr(p.arxiv_id, tldr_json_str)
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
            
    return get_block_html(
        title=p.title,
        authors=authors,
        rate=rate,
        arxiv_id=p.arxiv_id,
        tldr_abstract=p.tldr_zh,
        pdf_url=p.pdf_url,
        code_url=p.code_url,
        affiliations=affiliations_str,
        citation_count=p.citation_count,
        published_date=p.published_date,
        categories=p.categories,
        summary=p.summary,
        title_zh=p.title_zh
    )

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


