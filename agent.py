# %%
import os
import pathlib
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, tool, LiteLLMModel, CodeAgent
import urllib.parse
import requests
from bs4 import BeautifulSoup
import re
import json

# 환경 변수 로드
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / '.env')

# API 키 및 모델 설정
gemini_api_key = os.environ.get("GEMINI_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
claude_api_key = os.environ.get("CLAUDE_API_KEY")
hf_token = os.environ.get("HF_TOKEN")

gemini_model_id = "gemini/gemini-2.0-flash-lite"
openai_model_id = "gpt-4o-mini"
claude_model_id = "claude-3-5-sonnet-20241022"

# 기본 모델 (필요시 변경)
model = LiteLLMModel(model_id=gemini_model_id, api_key=gemini_api_key)

##############################
# 1. JD 수집용 scraping tools #
##############################

@tool
def scrape_indeed(job_title: str, location: str = "") -> str:
    """
    Indeed에서 실시간으로 JD를 수집합니다. 최신 Indeed 사이트 구조에 맞게 업데이트되었습니다.

    Args:
        job_title: 검색할 직무명
        location: 지역명 (선택)

    Returns:
        JSON 문자열 (최대 3개의 JD 정보)
    """
    try:
        query = urllib.parse.quote(job_title)
        loc = urllib.parse.quote(location) if location else ""
        url = f"https://www.indeed.com/jobs?q={query}&l={loc}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        
        for card in soup.select('div.job_seen_beacon')[:3]:
            link = card.select_one('h2.jobTitle a')
            job_url = "https://www.indeed.com" + link['href'] if link else ""
            
            title = link.get_text(strip=True) if link else ""
            
            company = card.select_one('span.companyName')
            company_name = company.get_text(strip=True) if company else ""
            
            snippet = card.select_one('div.job-snippet')
            jd_text = ' '.join([li.get_text(strip=True) for li in snippet.select('li')]) if snippet else ""
            
            results.append({
                "title": title,
                "company": company_name,
                "url": job_url,
                "jd": jd_text
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error scraping Indeed: {e}"

@tool
def scrape_glassdoor(job_title: str, location: str = "") -> str:
    """
    Glassdoor에서 실시간으로 JD를 수집합니다. 최신 Glassdoor 사이트 구조에 맞게 업데이트되었습니다.

    Args:
        job_title: 검색할 직무명
        location: 지역명 (선택)

    Returns:
        JSON 문자열 (최대 3개의 JD 정보)
    """
    try:
        query = urllib.parse.quote(job_title)
        loc = urllib.parse.quote(location) if location else ""
        url = f"https://www.glassdoor.com/Job/jobs.htm?suggestCount=0&suggestChosen=false&clickSource=searchBtn&typedKeyword={query}&sc.keyword={query}&locT=C&locId=1147401&jobType="
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.glassdoor.com/"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        
        for card in soup.select('li.react-job-listing')[:3]:
            job_url = "https://www.glassdoor.com" + card['data-job-url'] if 'data-job-url' in card.attrs else ""
            
            title = card.select_one('a[data-test="job-link"]').get_text(strip=True) if card.select_one('a[data-test="job-link"]') else ""
            
            company = card.select_one('div.d-flex.justify-content-between.align-items-start')
            company_name = company.get_text(strip=True) if company else ""
            
            snippet = card.select_one('div.job-description')
            jd_text = snippet.get_text(separator=' ', strip=True) if snippet else ""
            
            results.append({
                "title": title,
                "company": company_name,
                "url": job_url,
                "jd": jd_text
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error scraping Glassdoor: {e}"

@tool
def scrape_google_jobs(job_title: str, location: str = "") -> str:
    """
    Google 검색을 통해 JD를 수집합니다. 최신 Google Jobs 구조에 맞게 업데이트되었습니다.

    Args:
        job_title: 검색할 직무명
        location: 지역명 (선택)

    Returns:
        JSON 문자열 (최대 3개의 JD 정보)
    """
    try:
        query = urllib.parse.quote(f"{job_title} jobs {location}")
        url = f"https://www.google.com/search?q={query}&ibp=htl;jobs"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        
        for job in soup.select('div.ajG7l')[:3]:
            title = job.select_one('div.PwjeAc').get_text(strip=True) if job.select_one('div.PwjeAc') else ""
            
            company = job.select_one('div.nJlQNd.sMzDkb')
            company_name = company.get_text(strip=True) if company else ""
            
            link = job.select_one('a.pMhGee.Co68jc.j0vryd')
            job_url = link['href'] if link else ""
            
            snippet = job.select_one('div.YgLbBe.YRi0le')
            jd_text = snippet.get_text(separator=' ', strip=True) if snippet else ""
            
            results.append({
                "title": title,
                "company": company_name,
                "url": job_url,
                "jd": jd_text
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error scraping Google Jobs: {e}"

@tool
def scrape_linkedin(job_title: str, location: str = "") -> str:
    """
    LinkedIn에서 JD를 수집합니다. 최신 LinkedIn 사이트 구조에 맞게 업데이트되었습니다.

    Args:
        job_title: 검색할 직무명
        location: 지역명 (선택)

    Returns:
        JSON 문자열 (최대 3개의 JD 정보)
    """
    try:
        query = urllib.parse.quote(job_title)
        loc = urllib.parse.quote(location) if location else ""
        url = f"https://www.linkedin.com/jobs/search/?keywords={query}&location={loc}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "X-Requested-With": "XMLHttpRequest"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        
        for card in soup.select('div.base-card')[:3]:
            link = card.select_one('a.base-card__full-link')
            job_url = link['href'] if link else ""
            
            title = card.select_one('h3.base-search-card__title').get_text(strip=True) if card.select_one('h3.base-search-card__title') else ""
            
            company = card.select_one('h4.base-search-card__subtitle')
            company_name = company.get_text(strip=True) if company else ""
            
            snippet = card.select_one('div.base-search-card__info')
            jd_text = snippet.get_text(separator=' ', strip=True) if snippet else ""
            
            results.append({
                "title": title,
                "company": company_name,
                "url": job_url,
                "jd": jd_text
            })
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error scraping LinkedIn: {e}"

##############################
# 2. 빠른 웹서치 툴          #
##############################

@tool
def google_search(query: str) -> str:
    """
    Google 검색 결과 상위 3개를 반환합니다. 최신 Google 검색 구조에 맞게 업데이트되었습니다.

    Args:
        query: 검색할 쿼리 문자열

    Returns:
        검색 결과 3개의 제목, URL, 요약 텍스트를 포함한 문자열
    """
    try:
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        for g in soup.select('div.g')[:3]:
            title_elem = g.select_one('h3')
            title = title_elem.get_text(strip=True) if title_elem else ""
            link = g.select_one('a')
            href = link['href'] if link else ""
            snippet_elem = g.select_one('div.VwiC3b')
            snippet = snippet_elem.get_text(separator=' ', strip=True) if snippet_elem else ""
            results.append(f"{title}\n{href}\n{snippet}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error in google_search: {e}"

############################################
# 3. JD 수집용 WebAgent (ToolCallingAgent) #
############################################

web_jd_agent = ToolCallingAgent(
    tools=[scrape_indeed, scrape_glassdoor, scrape_google_jobs, scrape_linkedin, google_search],
    model=model,
    max_steps=8,
    name="web_jd_agent",
    description="""
    여러 구직 사이트와 검색엔진을 활용하여 최신 채용공고(JD)를 수집하는 에이전트.
    입력: {"job_title": "Data Scientist", "location": "Seoul"}
    출력: 상위 3개 JD (title, company, url, jd)
    """
)

web_jd_agent.prompt_templates["system_prompt"] += """
You are a Job Description (JD) Search Agent.

Given a job title and optional location, you must:
- Use all available tools (scrape_indeed, scrape_glassdoor, scrape_google_jobs, scrape_linkedin, google_search)
- Collect as many recent, relevant job descriptions as possible (at least 3)
- For each JD, extract:
    - Job title
    - Company name
    - URL
    - Short JD snippet or summary
- Return a JSON list of 3 best matching, recent JDs.

Output format:
[
  {"title": "...", "company": "...", "url": "...", "jd": "..."},
  {"title": "...", "company": "...", "url": "...", "jd": "..."},
  {"title": "...", "company": "...", "url": "...", "jd": "..."}
]
"""

############################################
# 4. ManagerAgent (JD 분석 및 매칭)        #
############################################

manager_agent = CodeAgent(
    tools=[],
    managed_agents=[web_jd_agent],
    model=model,
    name="manager_agent",
    description="JD 수집 및 사용자 CV와의 적합도 분석",
    max_steps=10,
    planning_interval=2,
    verbosity_level=2
)

manager_agent.prompt_templates["system_prompt"] += """
You are a Job Matching Assistant.

Your task:
- Input: User's CV text and target job title/location.
- Step 1: Use web_jd_agent to collect at least 3 recent, relevant JDs.
- Step 2: Analyze the user's CV against each JD.
- Step 3: Select the 3 most suitable JDs.
- Step 4: For each, write a detailed analysis covering:
    * Technical skills match
    * Experience relevance
    * Education fit
    * Quantifiable achievements
    * Career progression
    * Industry/domain knowledge
    * Company info (if available)
    * Potential contributions
    * Cultural fit
    * Growth potential
- Step 5: Return a JSON with 3 entries, each including:
    - company
    - job_url
    - detailed opinion (structured, markdown formatted)

Output format:
{
  "company_1": "...",
  "Job_url_1": "...",
  "Opinion 1": "...",
  "company_2": "...",
  "Job_url_2": "...",
  "Opinion 2": "...",
  "company_3": "...",
  "Job_url_3": "...",
  "Opinion 3": "..."
}
"""

##############################
# 5. Main 실행 함수          #
##############################

def main():
    import sys
    import json

    print("지원하고자 하는 직무명을 입력하세요:")
    job_title = input().strip()

    print("희망 근무 지역을 입력하세요 (없으면 엔터):")
    location = input().strip()

    print("CV 텍스트 파일 경로를 입력하세요:")
    cv_path = input().strip()

    if not os.path.exists(cv_path):
        print("CV 파일이 존재하지 않습니다.")
        sys.exit(1)

    with open(cv_path, 'r', encoding='utf-8') as f:
        cv_text = f.read()

    # ManagerAgent 실행
    prompt = f"""
User CV:
{cv_text}

Target Job Title: {job_title}
Location: {location}
"""
    print("JD 수집 및 분석 중... 잠시만 기다려주세요.")
    result = manager_agent.run(prompt)
    print("\n=== 최종 추천 결과 ===\n")
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except:
        print(result)

"""if __name__ == "__main__":
    main()"""

# %%
print(scrape_indeed(job_title="artificial intelligence", location=""))
print(scrape_glassdoor(job_title="artificial intelligence", location=""))
print(scrape_google_jobs(job_title="artificial intelligence", location="usa"))
print(scrape_linkedin(job_title="artificial intelligence", location="usa"))
print(google_search(query="artificial intelligence"))
# %%
