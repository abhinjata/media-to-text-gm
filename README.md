# media-to-text-gm
A self-improving media-to-text intelligence core with autonomous capabilities inspired by the theoretical architecture and algorithmic approach of the Gödel Machine.

## Implemented Architecture

Built on the back of a few core data classes, this specific implementation fo AI Agents was builty in a specific multi-agent framework. Throughout the program, you can find a few different vertices of AI Agents fulfilling their own function and then connected in a Graphical Strucutre with the help of LangGraph. 

Agents found: 

1) MediaTypeDetectorAgent, 2) TextExtractorAgent, 3) AudioTranscriberAgent, 4) VideoToAudioAgent, 5) UnifiedTextFormatterAgent, 6) SelfEvaluatorAgent, 7) ProposerAgent, 8) VerifierAgent

Allowing these agents to live inside their own nodes allows us to implement graphs as we usually would and have for decades, and this would also allow for a self-referential structure that is required for the self-improving recursive autonomous architecutre of the program.
