from typing import Optional, List, Any
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
import time
import json
from datetime import datetime, timedelta
from zotero_daily.llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken
from contextlib import ExitStack
from urllib.error import HTTPError
from abc import ABC, abstractproperty

from zotero_daily.citation_client import get_citation_count


class BasePaper(ABC):
    _score: Optional[float] = None
    _tldr_cache: Optional[str] = None
    _citation_count_cache: Optional[int] = None
    _citation_last_updated: Optional[str] = None
    storage: Optional['Storage'] = None
    disable_citation_check: bool = False

    @abstractproperty
    def title(self) -> str:
        pass
    @abstractproperty
    def summary(self) -> str:
        pass

    @abstractproperty
    def authors(self) -> List[any]:  # List of objects with .name attribute
        pass

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, value: float):
        self._score = value

    def set_tldr(self, tldr: str):
        self._tldr_cache = tldr

    @property
    def has_tldr(self) -> bool:
        return self._tldr_cache is not None

    @abstractproperty
    def arxiv_id(self) -> str:
        pass

    @abstractproperty
    def pdf_url(self) -> str:
        pass

    @abstractproperty
    def code_url(self) -> Optional[str]:
        pass

    @property
    def citation_count(self) -> Optional[int]:
        if self.disable_citation_check:
            return None

        # Rule 1: Don't fetch for papers published within the last 60 days.
        try:
            published = datetime.strptime(self.published_date, '%Y-%m-%d')
            if datetime.now() - published < timedelta(days=60):
                return None
        except (ValueError, TypeError):
            pass # Fallback for parsing errors

        # Rule 2: If citation count is fresh (less than 7 days old), return cached value.
        if self._citation_last_updated:
            try:
                last_updated = datetime.strptime(self._citation_last_updated, '%Y-%m-%d %H:%M:%S')
                if datetime.now() - last_updated < timedelta(days=7):
                    return self._citation_count_cache
            except (ValueError, TypeError):
                pass # Fallback for parsing errors

        # If rules don't apply, fetch new data.
        new_count = get_citation_count(self.title, self.authors)
        if new_count is not None and self.storage:
            self.storage.update_citation_count(self.arxiv_id, new_count)
            self._citation_count_cache = new_count
            self._citation_last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return self._citation_count_cache
    @property
    def title_zh(self) -> str:
        """Parses the TLDR JSON to get the Chinese title."""
        if not self.has_tldr:
            return ""
        try:
            data = json.loads(self.tldr)
            return data.get('title_zh', '')
        except (json.JSONDecodeError, TypeError):
            return ""

    @property
    def tldr_zh(self) -> str:
        """Parses the TLDR JSON to get the Chinese summary."""
        if not self.has_tldr:
            return ""
        try:
            data = json.loads(self.tldr)
            return data.get('tldr_zh', '')
        except (json.JSONDecodeError, TypeError):
            # Backwards compatibility: if it's not JSON, it's the old HTML string.
            return self.tldr

    @abstractproperty
    def tldr(self) -> str:
        pass

    @abstractproperty
    def affiliations(self) -> Optional[List[str]]:
        pass

    @abstractproperty
    def source(self) -> str:
        pass

    @abstractproperty
    def published_date(self) -> str:
        pass

    @abstractproperty
    def categories(self) -> List[str]:
        pass


class ArxivPaper(BasePaper):
    def __init__(self, paper: arxiv.Result, storage: Optional['Storage'] = None, disable_citation_check: bool = False):
        self._paper = paper
        self._score = None
        self._tldr_cache = None
        self._citation_count_cache = None
        self._citation_last_updated = None
        self.storage = storage
        self.disable_citation_check = disable_citation_check

    @property
    def source(self) -> str:
        return 'arxiv'

    @property
    def title(self) -> str:
        return self._paper.title

    @property
    def summary(self) -> str:
        return self._paper.summary

    @property
    def authors(self) -> List[Any]:  # objects returned by arxiv.Result.authors have .name
        return self._paper.authors

    @property
    def published_date(self) -> str:
        return self._paper.published.strftime('%Y-%m-%d')

    @property
    def categories(self) -> List[str]:
        return self._paper.categories


    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r'v\d+$', '', self._paper.get_short_id())

    @property
    def pdf_url(self) -> str:
        if self._paper.pdf_url is not None:
            return self._paper.pdf_url

        pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        if self._paper.links is not None:
            pdf_url = self._paper.links[0].href.replace('abs', 'pdf')

        ## Assign pdf_url to self._paper.pdf_url for pdf downloading (Issue #119)
        self._paper.pdf_url = pdf_url

        return pdf_url

    @cached_property
    def code_url(self) -> Optional[str]:
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(f'https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None

        if paper_list.get('count', 0) == 0:
            return None
        paper_id = paper_list['results'][0]['id']

        try:
            repo_list = s.get(f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None
        if repo_list.get('count', 0) == 0:
            return None
        return repo_list['results'][0]['url']

    @cached_property
    def tex(self) -> dict[str, str]:
        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            # file = self._paper.download_source(dirpath=tmpdirname)
            try:
                # 尝试下载源文件
                file = self._paper.download_source(dirpath=tmpdirname)
            except HTTPError as e:
                # 捕获 HTTP 错误
                if e.code == 404:
                    # 如果是 404 Not Found，说明源文件不存在，这是正常情况
                    logger.warning(f"Source for {self.arxiv_id} not found (404). Skipping source analysis.")
                    return None  # 直接返回 None，后续依赖 tex 的代码会安全地处理
                else:
                    # 如果是其他 HTTP 错误 (如 503)，这可能是临时性问题，值得记录下来
                    logger.error(f"HTTP Error {e.code} when downloading source for {self.arxiv_id}: {e.reason}")
                    raise  # 重新抛出异常，因为这可能是个需要关注的严重问题
            except Exception as e:
                logger.error(f"Error when downloading source for {self.arxiv_id}: {e}")
                return None
            try:
                tar = stack.enter_context(tarfile.open(file))
            except tarfile.ReadError:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: Not a tar file.")
                return None

            tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
            if len(tex_files) == 0:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                return None

            bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
            match len(bbl_file):
                case 0:
                    if len(tex_files) > 1:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple tex files while no bbl file.")
                        main_tex = None
                    else:
                        main_tex = tex_files[0]
                case 1:
                    main_name = bbl_file[0].replace('.bbl', '')
                    main_tex = f"{main_name}.tex"
                    if main_tex not in tex_files:
                        logger.debug(
                            f"Cannot find main tex file of {self.arxiv_id} from bbl: The bbl file does not match any tex file.")
                        main_tex = None
                case _:
                    logger.debug(
                        f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files.")
                    main_tex = None
            if main_tex is None:
                logger.debug(
                    f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")
            # read all tex files
            file_contents = {}
            for t in tex_files:
                f = tar.extractfile(t)
                content = f.read().decode('utf-8', errors='ignore')
                # remove comments
                content = re.sub(r'%.*\n', '\n', content)
                content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                # remove redundant \n
                content = re.sub(r'\n+', '\n', content)
                content = re.sub(r'\\\\', '', content)
                # remove consecutive spaces
                content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                if main_tex is None and re.search(r'\\begin\{document\}', content):
                    main_tex = t
                    logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                file_contents[t] = content

            if main_tex is not None:
                main_source: str = file_contents[main_tex]
                # find and replace all included sub-files
                include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}',
                                                                                          main_source)
                for f in include_files:
                    if not f.endswith('.tex'):
                        file_name = f + '.tex'
                    else:
                        file_name = f
                    main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
                file_contents["all"] = main_source
            else:
                logger.debug(
                    f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block.")
                file_contents["all"] = None
        return file_contents

    @property
    def tldr(self) -> str:
        if self._tldr_cache:
            return self._tldr_cache
            
        introduction = ""
        conclusion = ""
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            # remove cite
            content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
            # remove figure
            content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
            # remove table
            content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
            # find introduction and conclusion
            # end word can be \section or \end{document} or \bibliography or \appendix
            match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)',
                              content, flags=re.DOTALL)
            if match:
                introduction = match.group(0)
            match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)',
                              content, flags=re.DOTALL)
            if match:
                conclusion = match.group(0)
        llm = get_llm()
        response = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, providing the core idea to the user. Ensure the Chinese TLDR is more detailed and elaborates on the key findings.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        if "LLM generation skipped" in response:
            return response
        try:
            cleaned_response = response.replace('```json', '').replace('```', '').strip()
            json.loads(cleaned_response)  # Validate
            self._tldr_cache = cleaned_response
            return cleaned_response
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON response: {e}. Response: {response}")
            self._tldr_cache = response
            return response

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            # search for affiliations
            possible_regions = [r'\\author.*?\\maketitle', r'\\begin{document}.*?\\begin{abstract}']
            matches = [re.search(p, content, flags=re.DOTALL) for p in possible_regions]
            match = next((m for m in matches if m), None)
            if match:
                information_region = match.group(0)
            else:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: No author information found.")
                return None
            prompt = f"Given the author information of a paper in latex format, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]'. Following is the author information:\n{information_region}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
            prompt = enc.decode(prompt_tokens)
            llm = get_llm()
            affiliations = llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from the author information of a paper. You should return a python list of affiliations sorted by the author order, like ['TsingHua University','Peking University']. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            try:
                affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
                affiliations = eval(affiliations)
                affiliations = list(set(affiliations))
                affiliations = [str(a) for a in affiliations]
            except Exception as e:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: {e}")
                return None
            return affiliations


class SimpleAuthor:
    def __init__(self, name):
        self.name = name.strip()


class BioRxivPaper(BasePaper):
    def __init__(self, paper_data: dict, storage: Optional['Storage'] = None, disable_citation_check: bool = False):
        self._paper = paper_data
        self._score = None
        self._tldr_cache = None
        self._citation_count_cache = None
        self._citation_last_updated = None
        self.storage = storage
        self.disable_citation_check = disable_citation_check

    @property
    def source(self) -> str:
        return 'biorxiv'

    @property
    def title(self) -> str:
        return self._paper['title']

    @property
    def summary(self) -> str:
        return self._paper['abstract']

    @property
    def authors(self) -> List[SimpleAuthor]:
        raw = self._paper.get('authors', '')
        # Split by semicolon
        names = raw.split(';')
        return [SimpleAuthor(n) for n in names if n.strip()]

    @property
    def arxiv_id(self) -> str:
        return self._paper['doi']

    @property
    def pdf_url(self) -> str:
        # Construct PDF URL: https://www.biorxiv.org/content/10.1101/2023.03.20.533581v1.full.pdf
        return f"https://www.biorxiv.org/content/{self._paper['doi']}v1.full.pdf"

    @property
    def code_url(self) -> Optional[str]:
        return None

    @property
    def tldr(self) -> str:
        if self._tldr_cache:
            return self._tldr_cache

        llm = get_llm()
        prompt = """Given the title and abstract of a paper, generate a JSON object with three keys:
"title_zh": Translate the title to Chinese.
"tldr_en": A one-sentence TLDR summary in English.
"tldr_zh": A one-sentence TLDR summary in Chinese.

Strictly return ONLY the JSON object, no markdown formatting.

Title: {title}
Abstract: {abstract}
"""
        prompt = prompt.format(
            title=self.title,
            abstract=self.summary
        )

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        prompt = enc.decode(prompt_tokens)

        response = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, providing the core idea to the user. Ensure the Chinese TLDR is more detailed and elaborates on the key findings.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        if "LLM generation skipped" in response:
            return response
        try:
            cleaned_response = response.replace('```json', '').replace('```', '').strip()
            json.loads(cleaned_response)  # Validate
            self._tldr_cache = cleaned_response
            return cleaned_response
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON response: {e}. Response: {response}")
            self._tldr_cache = response
            return response

    @property
    def affiliations(self) -> Optional[List[str]]:
        return None

    @property
    def published_date(self) -> str:
        return self._paper.get('date', 'N/A')

    @property
    def categories(self) -> List[str]:
        return [c.strip() for c in self._paper.get('category', '').split(';')]
