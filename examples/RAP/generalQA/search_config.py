import io
import re
from typing import TypedDict, Optional
import numpy as np
from reasoners import SearchConfig, LanguageModel
from parsers import self_eval_parser
from world_model import GeneralRAPState, GeneralRAPAction, GeneralRAPPromptDict
import pdb
class GeneralRAPUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str

class GeneralRAPConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 useful_prompt: GeneralRAPUsefulPrompt,
                 n_actions=4,
                 batch_size=1,
                 temperature=0.8,
                 top_k=50,
                 top_p=0.95,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.useful_prompt = useful_prompt
        self.example = ''
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        self.overall_question: Optional[str] = None
        self.prompt_instruction = ""   
        self.n_shots = 0

    def update_example(self, example: str, prompt: GeneralRAPPromptDict = None) -> None:
        super().update_example(example, prompt=prompt)

        assert prompt is not None
        self.prompt = prompt
        self.prompt_instruction = self.prompt['instruction'] + '\n\n'
        # with io.StringIO() as f:
        #     f.write(self.prompt['instruction'] + '\n\n')
        #     for idx, example in enumerate(self.prompt['interactive_examples']):
        #         f.write(example.format(idx=idx + 1) + '\n\n')
        #     self.n_shots = len(self.prompt['interactive_examples'])
        #     self.prompt_instruction = f.getvalue()

        # if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            # print(self.example)
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example, flags=re.DOTALL)[1]
            # self.overall_question = re.match('.*((([A-Z].* (calculate|how|what|find|true or false))|((Calculate|How|What|Find|True or false))).*)$', self.example, flags=re.DOTALL)[1]
        self.overall_question = self.example

    def get_actions(self, state: GeneralRAPState, ) -> list[GeneralRAPAction]:
        with io.StringIO() as f:
            f.write(self.prompt_instruction)
            f.write(self.prompt["question_prefix"].format(idx=self.n_shots + 1, question=self.example) + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1))
            if at_depth_limit := self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" " + self.prompt["overall_question_prefix"])
            model_input = f.getvalue()

        # print(model_input)
        # print("-"*100)

        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0 if at_depth_limit else self.temperature
        outputs = []
        system_prompt = """
        <Task Instructions>
        You must generate a subquestion that will help you answer the overall question.
        The preceding examples are provided for context.
        You main focus should be the last question. 
        Make sure to think hard about what subquestion is needed to understand the overall question better and answer it.
        Each subquestion should help lead to closer to answering the overall question.
        Feel free to self-reflect and rethink your approach if you feel like you made a mistake or are going in the wrong direction.

        Occasionally, ask yourself reflection questions to ensure you are on the right track.
        If you feel you are off course, ask yourself questions to help you get back on track.
        For example: What assumptions am I making about this problem that need to be questioned?, What are the underlying principles or concepts related to this question that I need to understand better?, How can I break this problem into smaller, more manageable parts to analyze it effectively?, What data or evidence is necessary to answer this question, and how can I obtain it?, What alternative perspectives or explanations could challenge my initial understanding of this question?, What would the answer look like if one or more of my assumptions were incorrect?, What are the historical or contextual factors that influence the dynamics of this question?, How does this question connect to broader concepts or systems I am already familiar with?, What potential biases might affect my approach to answering this question, and how can I mitigate them?, If I were to explain this question to someone else, what knowledge or context would they need to understand it fully?, If I were to explain this question to someone else, what knowledge or context would they need to understand it fully?

        Approach every problem - whether it is mathematics, code, or knowledge of our world - with genuine curiosity and doubt.
        Before settling on any answer, question your own assumptions, explore different paths of thought, and always seek deeper truth.
        After every 3rd subquestion, ask yourself a self-reflection question or think of how you can improve your approach.

        Important: Never repeat a subquestion.

        <Task Instructions>
        <Output Format>
            Output should be plain text sentences with the subquestion. There should be no prefix or suffix.
            Just the subquestion.
        </Output Format>
        """
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            outputs += self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                top_k=self.top_k,
                                                top_p=self.top_p,
                                                # eos_token_id='\n',
                                                system_prompt=system_prompt).text
            
        # print(outputs)
        outputs = [output.strip() for output in outputs]
        if at_depth_limit:
            outputs = [self.prompt["overall_question_prefix"] + ' ' + output for output in outputs]
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.prompt["overall_question_prefix"] in output:
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                if self.overall_question.lower() == output.lower():
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        outputs = list(dict.fromkeys(outputs))
        print("-"*100)
        print("Subquestions:", outputs)
        return outputs

    def fast_reward(self, state: GeneralRAPState, action: GeneralRAPAction) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, _, _) in enumerate(state):
                f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
            f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.useful_prompt["useful_prefix"])
            model_input = f.getvalue()
        
        # pdb.set_trace()

        # logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        # probs = np.exp(logits) / np.sum(np.exp(logits))
        # useful_prob = probs[0]
        # fast_reward, _ = self.calculate_reward(useful_prob)
        print("-"*100)
        print("Model Input For Evaluation:", model_input)
        num_retry = 10
        success = False
        while num_retry > 0 and not success:
            try:
                system_prompt = f"""
                <Task Instructions>
                You must evaluate whether the subquestion is useful to answer the overall question.
                The preceding examples are provided for context.
                The score you give should be based on two main factors:
                    1. Does this question help in leading to the answer of the overall question and is it a high quality question?
                        - This should be scored between 0 and 3:
                            - 0: This question does not help in leading to the answer of the overall question or is not a high quality question.
                            - 3: This question significantly helps in leading to the answer of the overall question and is a high quality question.
                    2. Are there better subquestions that could be asked?
                        - This should be scored between 0 and 7:
                            - 0: This is a poor subquestion.
                            - 7: This is the best possible subquestion.

                If the question helps lead to the answer and is the best subquestion, then give it a score of 10.

                If there are no more subquestions after the provided subquestion that can be asked, then give it a score of 10.
                
                The total score should be a weighted combination of the above two factors. 
                Make sure to think step-by-step and justify your scores.
                </Task Instructions>
                <Output Format>
                {self_eval_parser.get_format_instructions()}
                </Output Format>
                """
                self_eval = self.base_model.generate([model_input], 
                                                system_prompt=system_prompt)
                self_eval = self_eval_parser.parse(self_eval.text[0])
                self_eval_score = self_eval['self_eval_score'] / 10

                for key, value in self_eval.items():
                    print(f"{key}: {value}")

                success = True
            except Exception as e:
                print(e)
                num_retry -= 1
        print("-"*100)
        if not success:
            print("Warning: no self eval found")
            self_eval_score = 5
                
        return self_eval_score, {'r_useful': self_eval_score}

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}

    def reward(self, state: GeneralRAPState, action: GeneralRAPAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        # return confidence, {'r_conf': confidence}
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)