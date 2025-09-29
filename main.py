import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


class FallTemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    This bot is what is used by Metaculus in our benchmark, but is also provided as a template for new bot makers.

    This template is given as-is, and though we have covered most test cases
    in forecasting-tools it may be worth double checking key components locally.

    Main changes since Q2:
    - An LLM now parses the final forecast output (rather than programmatic parsing)
    - Added resolution criteria and fine print explicitly to the research prompt
    - Previously in the prompt, nothing about upper/lower bound was shown when the bounds were open. Now a suggestion is made when this is the case.
    - Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)

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

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ones.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLMto intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions for the
    MiniBench and Seasonal AIB tournaments. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknewsasknews/deep-research/low":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "model_name").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """


    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            try:
                research = await self._call_perplexity(question.question_text)
            except Exception as e:
                logger.error(f"Perplexity call failed for {question.page_url}: {e}")
                research = ""

            if not research.strip():
                logger.warning(f"No research returned for {question.page_url}. Continuing anyway.")
            else:
                logger.info(f"Found Research for URL {question.page_url}:\n{research}")

            return research


    async def _call_perplexity(self, question: str) -> str:
        """
        Query Perplexity's sonar-reasoning-pro model for a detailed research report.
        Uses your custom prompt, with error handling and logging.
        """
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.

            Your job is to generate a concise but detailed summary of the most relevant and recent information, including:
            - Key events, entities, or developments related to the question
            - Relevant historical context or trends
            - Current news reports, announcements, or data that may influence the outcome
            - Any uncertainty, missing information, or upcoming events that could affect the resolution
            - Arguments or evidence that support each possible resolution (e.g. Yes or No)

            **If you don't find recent news, that is surprising — double check**

            Check for similar questions that have been asked on prediction markets:
            - www.polymarket.com
            - www.kalshi.com
            - www.predictit.org/markets
            - www.metaculus.com/questions/

            Note any differences between those markets and this question, such as resolution dates or criteria.

            If the question could currently be resolved based on available information, say so clearly, with supporting evidence.

            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )

        model = GeneralLlm(
            model="perplexity/sonar-reasoning-pro",
            temperature=0.1,
        )

        try:
            response = await model.invoke(prompt)
            if not response.strip():
                logger.warning("Perplexity returned empty response.")
                return "No research found."
            return response
        except Exception as e:
            logger.error(f"Error calling Perplexity: {e}")
            return "Research unavailable due to API error."

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

            (a) Time remaining until resolution and key upcoming milestones
            (b) The status quo outcome if nothing changed from today
            (c) Scope sensitivity: How would your forecast change if the timeframe were shorter/longer?
            (d) Reference class: What is the historical base rate for similar events?
            (e) Scenario for NO outcome
            (f) Scenario for YES outcome
            (g) Key uncertainties
            (h) Confidence check: Write a brief note to your future self explaining your reasoning

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        logger.info(f"Forecasted URL {question.page_url} with prediction: {decimal_pred}")
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

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

            (a) Time remaining until resolution and key upcoming milestones
            (b) The status quo outcome if nothing changed from today
            (c) Reference class: What is the historical distribution for similar events?
            (d) Most likely outcome and why
            (e) Second most likely outcome and why  
            (f) Key factors that could cause an upset
            (g) Uncertainty assessment

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with "Option", remove it if not part of the name.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}")
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
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

            Units for answer: {question.unit_of_measure or "Not stated (please infer this)"}

            You have two research assistants, they have compiled this report:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Your 80% confidence interval (10th to 90th percentile) should contain the true value 8 times out of 10 for similar questions.

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
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        upper_bound_number = (
            question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        )
        lower_bound_number = (
            question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        )

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number}."

        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "parser": "openai/gpt-4o-mini",
        },
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)

