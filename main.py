import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)


class TemplateForecaster(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = 2  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""

            # First: AskNews research
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                asknews_research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
                if asknews_research.strip():
                    research += asknews_research


            # Second: Perplexity research
            perplexity_research = ""
            if os.getenv("PERPLEXITY_API_KEY"):
                perplexity_research = await self._call_perplexity(question.question_text)
                
            # Combine the two research reports
            if perplexity_research.strip():
                if research:  # If we got AskNews research
                    research += "\n\n" + "="*50 + "\n"
                    research += "# START OF SECOND RESEARCHER'S REPORT\n"
                    research += "="*50 + "\n\n"
                research += perplexity_research
        
             # Fail fast if both research sources failed
            if not research.strip():
                raise Exception(f"Both AskNews and Perplexity research failed for question {question.page_url}. Script will retry in 30 minutes.")
        
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research


    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
COMBINED_ENHANCED_RESEARCH_PROMPT = """
            You are a research assistant to a world-class superforecaster.

            The superforecaster will give you a forecasting question they intend to predict. Your job is to generate a comprehensive research report that will help them make the most accurate forecast possible.

            **QUESTION TO RESEARCH:** {question}

            **RESEARCH PRIORITIES (in order of importance):**
            1. Current status and recent developments (last 3-6 months)
            2. Historical precedents and base rates for similar events
            3. Key stakeholders and their stated positions/incentives  
            4. Upcoming milestones, deadlines, or decision points
            5. Leading indicators that could signal changes
            6. Expert opinions and prediction market prices
            7. Quantitative data trends where available

            **REQUIRED RESEARCH STRUCTURE:**

            **Executive Summary** (2-3 sentences on current status)
            - What is the current state regarding this question?
            - Can this question be resolved NOW based on available information? If yes, state clearly with evidence.

            **Recent Developments** (chronological, most recent first)
            - Key events from the last 3-6 months directly relevant to the question
            - Include specific dates, numbers, and quotes where relevant
            - Distinguish between facts, claims, and speculation
            
            **Historical Context and Base Rates**
            - Similar past events and their outcomes
            - What percentage of similar events typically resolve positively?
            - Relevant trends over time
            
            **Key Stakeholders Analysis**
            - Who are the main decision-makers or influencers?
            - What are their stated positions, incentives, and likely actions?
            - Any recent statements or commitments?

            **Prediction Market Check**
            Check for similar questions on prediction markets:
            - www.polymarket.com
            - www.kalshi.com  
            - www.predictit.org/markets
            - www.metaculus.com/questions/
            - Other betting markets or expert forecasts

            Note the similarities and differences between these markets and the question asked (e.g. differing resolution dates, conditions, or scope).
            
            **Leading Indicators and Upcoming Events**
            - What events, announcements, or data releases could affect the outcome?
            - Specific dates and deadlines to watch
            - Early warning signals that might predict the outcome
            
            **Arguments For Each Resolution**
            - **Arguments for YES/positive resolution:** What evidence and factors support this outcome?
            - **Arguments for NO/negative resolution:** What evidence and factors support this outcome?
            
            **Critical Uncertainties and Information Gaps**
            - What key information is missing or unknown?
            - What could dramatically change your assessment?
            - Note any conflicting information or expert disagreement
            
            **QUALITY STANDARDS:**
            - Focus ONLY on information directly relevant to this specific question
            - Prioritize recent, authoritative sources over older or speculative ones
            - Include specific dates, numbers, and direct quotes where relevant
            - If you find very little recent news about this topic, explicitly note this as significant
            - **CRITICAL:** If the question could currently be resolved based on available information, state this clearly with supporting evidence
            
            **RESEARCH SCOPE WARNING:**
            If you cannot find sufficient relevant information about this specific question, explicitly state: "Limited relevant information found for this specific question." Do not include unrelated news just to fill space.
            
            **FINAL CHECK:**
            Before submitting, verify that your research directly addresses the forecasting question and would help a superforecaster make an informed prediction.
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-reasoning-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around an search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional superforecaster with a track record of exceptional accuracy.
            
            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            You have two research assistants, they have compiled this report:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Use this structured approach:

            **STRUCTURED ANALYSIS:**
            (a) Time remaining until resolution and key upcoming milestones
            (b) The status quo outcome if nothing changed from today
            (c) Scope sensitivity: How would your forecast change if the timeframe were shorter/longer?
            (d) Reference class: What is the historical base rate for similar events?
            (e) Scenario for NO outcome (most likely path)
            (f) Scenario for YES outcome (what would need to change)
            (g) Key uncertainties and information that could shift your view
            (h) Confidence check: Write a brief letter to your future self explaining your reasoning in case you're wrong

            **FORECASTING PRINCIPLES:**
            - The world changes slowly most of the time (status quo bias)
            - Extraordinary claims require extraordinary evidence
            - Consider both inside view (specific details) and outside view (reference class)
            - Account for regression to the mean
            - Be humble about your uncertainty

            **FINAL CALIBRATION:**
            Think about similar questions you've seen. If you gave this probability to 100 similar questions, how many should resolve positively to be well-calibrated.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional superforecaster with a track record of exceptional accuracy.
            
            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            You have two research assistants, they have compiled this report:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Use this structured approach:

            **STRUCTURED ANALYSIS:**
            (a) Time remaining until resolution and key upcoming milestones
            (b) The status quo outcome if nothing changed from today
            (c) Reference class: What is the historical distribution for similar events?
            (d) Most likely outcome and why
            (e) Second most likely outcome and why  
            (f) Key factors that could cause an upset/unexpected outcome
            (g) Uncertainty assessment: Which options are you most/least confident about?

            **FORECASTING PRINCIPLES:**
            - The world changes slowly most of the time (favor status quo)
            - Leave meaningful probability on unexpected outcomes (avoid overconfidence)
            - Consider both inside view (specific details) and outside view (reference class)
            - Probabilities should sum to 100%
            - Don't put 0% on plausible outcomes

            **SANITY CHECKS:**
            - Do your probabilities reflect your actual confidence levels?
            - Have you been appropriately humble about uncertainty?
            - Are you giving the status quo option enough weight?

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional superforecaster with a track record of exceptional accuracy.
            
            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            You have two research assistants, they have compiled this report:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Use this structured approach:

            **STRUCTURED ANALYSIS:**
            (a) Time remaining until resolution and key upcoming milestones
            (b) The status quo/baseline outcome if current trends continued
            (c) Reference class: What is the historical range for similar metrics?
            (d) Key factors that could drive the outcome higher than baseline
            (e) Key factors that could drive the outcome lower than baseline  
            (f) Most likely scenario (median outcome)
            (g) Tail risk scenarios (10th and 90th percentile outcomes)

            **FORECASTING PRINCIPLES:**
            - The world changes slowly most of the time (favor continuation of trends)
            - Be humble: Set wide confidence intervals to account for unknown unknowns
            - Consider both inside view (specific details) and outside view (reference class)
            - Regression to the mean is powerful for extreme values
            - Extraordinary outcomes require extraordinary evidence

            **UNCERTAINTY CALIBRATION:**
            Your 80% confidence interval (10th to 90th percentile) should contain the true value 8 times out of 10 for similar questions. Be appropriately humble.
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
         llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
             "default": GeneralLlm(
                 model="openrouter/openai/gpt-4o", # openai/gpt-4o # google/gemini-2.5-pro-preview
                 temperature=0.3,
                 timeout=40,
                 allowed_tries=2,
             ),
             "summarizer": "openai/gpt-4o-mini",
         },
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
