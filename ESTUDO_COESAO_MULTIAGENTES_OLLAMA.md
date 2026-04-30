# Estudo Inicial: Coesao de Redacao, Multiagentes e Ollama

## 1. Objetivo deste estudo

Este documento foi preparado para apoiar uma tarefa de analise sobre:

- os repositorios `RevisaoSistematica` e `RevisaoTecnica`
- o uso de multiagentes no servidor atual
- o uso de Ollama como infraestrutura local de modelos
- a necessidade de melhorar a coesao entre capitulos e secoes sem perder qualidade tecnica
- a viabilidade de um router avancado com designacao de tarefas e memoria compartilhada

O texto foi escrito em nivel introdutorio, pensando em quem ainda esta aprendendo o tema.

## 2. Resumo executivo

Os dois repositorios estao em niveis arquiteturais diferentes.

`RevisaoSistematica` ja possui uma base relativamente avancada: ha pipeline tradicional, camada multiagente, blackboard compartilhado, revisao iterativa de secoes, controle de conceitos repetidos e pos-processamento por capitulo. Mesmo assim, a coesao global ainda sofre porque boa parte da escrita continua acontecendo por secoes quase independentes, e a harmonizacao narrativa entra tarde, depois que os textos ja foram produzidos.

`RevisaoTecnica` e mais simples e linear: ele faz scraping, avaliação com Ollama e geração de relatório. Hoje ele não possui arquitetura multiagente, memoria compartilhada, planejamento global de narrativa nem revisao iterativa entre especialistas. Para analise de patentes, isso pode ser suficiente. Para redação longa, coesa e multi-capitulo, não.

O problema central, portanto, nao e apenas "falta de LLM". O problema e de orquestracao narrativa:

- quem decide o papel de cada secao dentro do capitulo
- como cada agente sabe o que ja foi dito
- como evitar repeticao e contradicao entre secoes
- como manter o mesmo fio argumentativo do inicio ao fim

Minha conclusao e:

1. O servidor atual tem capacidade real para uma arquitetura melhor.
2. `RevisaoSistematica` ja possui pecas importantes, mas ainda sem um router narrativo forte.
3. O proximo passo mais promissor nao e "mais agentes" apenas; e combinar:
   - supervisor
   - router hibrido
   - designacao explicita de papeis
   - memoria compartilhada de estado narrativo
   - revisao em nivel de capitulo, nao so em nivel de secao

## 3. O que foi encontrado no servidor

### 3.1 Ollama

O Ollama esta ativo localmente em `http://localhost:11434`, o que esta alinhado com a configuracao do projeto.

Modelos instalados no servidor:

- `qwen3:8b`
- `mixtral:8x22b`
- `qwen2.5:32b`
- `qwen2.5:14b`
- `gemma3:12b`
- `gemma3:4b`
- `llama3.1:70b`

No momento da verificacao, nao havia modelo carregado em memoria (`/api/ps` retornou lista vazia), o que e bom para evitar VRAM ocupada sem uso.

### 3.2 GPU

O servidor possui:

- `NVIDIA GeForce RTX 5090`
- cerca de `32 GB` de VRAM

Isso significa que o ambiente e adequado para:

- inferencia local com modelos medios e grandes quantizados
- pipelines multiagente moderados
- roteamento com modelos menores e escrita/revisao com modelos melhores

Em termos práticos: o gargalo principal nao parece ser hardware. O gargalo principal parece ser desenho de fluxo.

## 4. Diagnostico por repositorio

### 4.1 RevisaoSistematica

Pontos fortes observados:

- possui pipeline completo de revisao sistematica
- possui camada multiagente em `src/agents/`
- usa blackboard compartilhado
- tenta evitar redundancia com `ConceptRegistry`
- faz revisao iterativa entre escrita e avaliacao
- possui pos-processamento para coerencia e deduplicacao por capitulo
- usa Ollama por API local

Arquivos centrais:

- `src/review_writer.py`
- `src/post_processor.py`
- `src/agents/coordinator_agent.py`
- `src/agents/blackboard.py`
- `src/agents/writing_agent.py`
- `src/agents/review_agent.py`
- `config/config.yaml`
- `config/config_5090.yaml`

Pontos fracos para coesao:

- a escrita ainda nasce principalmente por secao
- o registro de conceitos ajuda contra repeticao conceitual, mas nao garante progressao argumentativa
- a memoria compartilhada atual guarda estado, mas ainda nao funciona como memoria narrativa forte
- a revisao por capitulo existe, mas entra depois da geracao principal
- o sistema ainda depende demais de "consertar depois" em vez de "planejar antes"
- o `multi_agent.enabled` em `config/config_5090.yaml` esta `false`, entao a arquitetura mais rica pode nem estar sendo usada no fluxo principal

### 4.2 RevisaoTecnica

Pontos fortes observados:

- arquitetura simples e facil de entender
- bom para scraping, avaliacao inicial e geracao de relatorio
- integracao direta com Ollama
- fluxo curto e objetivo

Arquivos centrais:

- `main.py`
- `evaluator/llm_evaluator.py`
- `report/generator.py`
- `config.py`

Limitacoes:

- nao ha multiagentes
- nao ha supervisor
- nao ha roteamento avancado
- nao ha memoria compartilhada de longo prazo
- nao ha planejamento global de narrativa
- nao ha loop de escrita -> revisao -> reescrita

Conclusao: `RevisaoTecnica` e bom como pipeline funcional de busca e avaliacao. Nao e, no estado atual, uma base suficiente para narrativa tecnica longa com coesao entre capitulos.

## 5. Problema central da coesao de redacao

Em linguagem simples: hoje o sistema sabe gerar texto tecnico, mas ainda nao sabe "conduzir uma historia argumentativa longa" de forma confiavel.

Isso aparece em quatro tipos de problema:

### 5.1 Secoes corretas, mas pouco conectadas

Cada secao pode estar tecnicamente aceitavel isoladamente, mas o capitulo como um todo pode ficar sem fluxo natural.

Exemplo de sintoma:

- a secao B nao responde nem desenvolve o que a secao A abriu
- o capitulo parece uma colecao de mini-relatorios, nao um argumento unico

### 5.2 Repeticao semantica

O mesmo ponto aparece em secoes diferentes, com palavras diferentes.

O `ConceptRegistry` reduz repeticao de termos/conceitos explicados, mas nao resolve sozinho:

- repeticao de conclusoes
- repeticao de ressalvas
- repeticao de contexto
- repeticao de tabelas ou evidencias muito parecidas

### 5.3 Falta de memoria narrativa

Existe memoria de execucao, mas ainda falta uma memoria orientada a redacao, por exemplo:

- qual tese do capitulo ja foi estabelecida
- qual lacuna ficou aberta na secao anterior
- qual ponto a proxima secao precisa desenvolver
- quais afirmacoes ja foram usadas como eixo argumentativo

### 5.4 Roteamento bom para evidencia, fraco para funcao discursiva

O sistema atual filtra chunks e evidencias por tema e relevancia. Isso e importante.

Mas escrever bem um capitulo exige uma segunda pergunta:

"Qual e a funcao discursiva desta secao dentro do capitulo?"

Exemplos de funcao discursiva:

- definir conceito base
- comparar abordagens
- discutir controversias
- consolidar consenso
- explicar limitacoes
- preparar a transicao para o proximo topico

Hoje o sistema olha mais para "evidencia relevante" do que para "papel narrativo da secao".

## 6. Evidencias concretas de que a coesao ainda e um problema

A partir da saida atual de `RevisaoSistematica`, aparecem sinais claros:

- titulo final excessivamente agregador e pouco natural
- repeticao forte de tabelas ou estruturas muito similares entre secoes
- partes da agenda de pesquisa saindo em ingles
- mistura de evidencias economicas ou de metanol em secoes que deveriam estar mais restritas a armazenamento frio/CO2
- secoes com bom nivel tecnico local, mas ainda pouco articuladas entre si

Ou seja: o sistema nao esta "ruim". Ele esta em um estagio intermediario:

- bom para gerar conteudo
- razoavel para revisar
- ainda insuficiente para garantir unidade argumentativa forte

## 7. Estudos que devem entrar na lista da tarefa

Segue uma lista organizada e didatica do que estudar.

### 7.1 Estudo do repositorio RevisaoSistematica

O que estudar:

- pipeline atual
- escrita por secao
- mecanismo de tese por capitulo
- `ConceptRegistry`
- `blackboard`
- loop `writing -> review`
- pos-processamento de coerencia e deduplicacao

Pergunta-chave:

"Por que, mesmo com esses mecanismos, a coesao global ainda nao fica forte o suficiente?"

### 7.2 Estudo do repositorio RevisaoTecnica

O que estudar:

- fluxo de scraping
- avaliacao com Ollama
- geracao de relatorio
- ausencia de planejamento narrativo e multiagente

Pergunta-chave:

"Quais partes deste repositorio podem ser reaproveitadas como agentes especialistas, mas nao como orquestrador principal?"

### 7.3 Estudo de multiagentes aplicados neste servidor

O que estudar:

- supervisor
- workers especialistas
- troca de mensagens
- compartilhamento de estado
- custo de paralelizacao
- efeito do paralelismo na qualidade textual

Pergunta-chave:

"Quando usar varios agentes melhora a qualidade, e quando apenas aumenta complexidade e latencia?"

### 7.4 Estudo de Ollama

O que estudar:

- API local
- `generate` e `chat`
- uso de `format=json` e schema estruturado
- `keep_alive`
- telemetria basica de execucao
- embeddings locais
- escolha de modelos por funcao

Pergunta-chave:

"Como usar Ollama de forma mais arquitetural, e nao so como um endpoint de texto?"

### 7.5 Estudo de router avancado

Este ponto deve entrar explicitamente na lista, como voce pediu.

O que estudar:

- router deterministico + LLM
- classificar a requisicao por dominio e por funcao discursiva
- decidir qual agente escreve, qual revisa e qual sintetiza
- decidir quando paralelizar e quando sequenciar

Pergunta-chave:

"Como fazer o sistema mandar a tarefa certa para o agente certo, na ordem certa?"

### 7.6 Estudo de designacao de tarefas

"Designacao" aqui significa atribuicao explicita de responsabilidade.

O que estudar:

- qual agente e dono de cada subproblema
- quem pode escrever
- quem pode revisar
- quem pode consolidar memoria
- quem pode aprovar transicoes entre secoes

Pergunta-chave:

"Como evitar que todos os agentes tentem fazer tudo ao mesmo tempo?"

### 7.7 Estudo de memoria compartilhada

Este e um ponto central.

O que estudar:

- blackboard atual
- memoria de conceitos
- memoria de capitulo
- memoria de secoes aprovadas
- memoria de decisoes narrativas
- memoria de lacunas e contradicoes

Pergunta-chave:

"Como garantir que o sistema lembre nao apenas fatos, mas o estado da argumentacao?"

## 8. Proposta de arquitetura recomendada

Minha recomendacao nao e abandonar o que existe. E evoluir o que ja existe.

### 8.1 Ideia central

Passar de:

- "escrita por secao com revisao posterior"

para:

- "planejamento de capitulo -> roteamento -> escrita designada -> revisao local -> harmonizacao de capitulo -> montagem final"

### 8.2 Arquitetura recomendada em camadas

#### Camada 1: Supervisor central

Responsabilidade:

- entender o objetivo global
- quebrar o trabalho por capitulo e secao
- decidir a ordem das etapas
- chamar os especialistas certos

Esse supervisor nao deve escrever tudo.
Ele deve coordenar.

#### Camada 2: Router avancado

Responsabilidade:

- classificar a tarefa por dominio tecnico
- classificar a tarefa por funcao narrativa
- designar o melhor agente para cada etapa

Exemplo de funcoes narrativas:

- contextualizacao
- comparacao
- controversia
- sintese
- limitacoes
- transicao

Recomendacao pratica:

- usar um router hibrido
- parte deterministica para regras obvias
- parte LLM para casos ambíguos

#### Camada 3: Memoria compartilhada

Responsabilidade:

- registrar o que cada capitulo quer provar
- registrar o que cada secao ja afirmou
- registrar conceitos ja explicados
- registrar contradicoes detectadas
- registrar decisoes de estilo e terminologia
- registrar pendencias abertas para a proxima secao

Memoria recomendada em 3 niveis:

1. Memoria global da revisao
2. Memoria por capitulo
3. Memoria por secao

#### Camada 4: Agentes especialistas

Sugestao de especializacao:

- agente de planejamento de capitulo
- agente de roteamento/designacao
- agente de escrita de secao
- agente de revisao tecnica
- agente de revisao de coesao
- agente de harmonizacao final de capitulo

O sistema atual ja tem parte disso, mas ainda falta separar melhor:

- revisao tecnica
- revisao narrativa
- consolidacao de memoria

#### Camada 5: Harmonizador de capitulo

Este ponto e o mais importante para resolver sua dor principal.

Em vez de depender apenas de revisao local de cada secao, ter um agente final de capitulo com funcao explicita de:

- remover redundancias entre secoes
- alinhar terminologia
- reforcar a tese do capitulo
- criar transicoes naturais
- garantir progressao logica

## 9. Como isso se encaixa no servidor atual

O servidor atual suporta bem uma arquitetura desse tipo porque:

- ha GPU forte
- ha varios modelos locais instalados
- o Ollama esta funcional
- `RevisaoSistematica` ja possui base multiagente e blackboard

Em outras palavras:

- o problema principal nao e infraestrutura
- o problema principal e desenho de orquestracao

## 10. Recomendacao pratica de uso dos modelos

Sem entrar em afinacoes muito avancadas, uma estrategia inicial plausivel seria:

- router/supervisor: modelo medio e rapido
- escrita principal: modelo mais forte
- revisao de coesao: modelo mais forte
- extracao/rotinas estruturadas: modelo medio com schema JSON

No servidor atual, uma divisao inicial razoavel seria:

- `qwen2.5:14b` para escrita e revisao
- `gemma3:4b` ou `qwen3:8b` para classificacao/roteamento rapido
- `qwen2.5:32b` apenas quando o ganho justificar o custo

Observacao importante:

Se houver paralelismo, ele precisa ser calibrado com cuidado. Mais paralelismo pode piorar coesao se varias secoes forem escritas ao mesmo tempo sem memoria narrativa consistente.

## 11. Recomendacao de evolucao em fases

### Fase 1: Melhorar sem refatoracao grande

Objetivo:

- melhorar coesao com baixo risco

Acoes:

- ativar de forma controlada o fluxo multiagente
- fortalecer a memoria de capitulo
- adicionar "papel narrativo da secao" ao prompt
- separar revisao tecnica de revisao de coesao
- reforcar harmonizacao final por capitulo

### Fase 2: Introduzir router avancado com designacao

Objetivo:

- fazer cada tarefa ir ao agente correto

Acoes:

- criar classificador de dominio
- criar classificador de funcao discursiva
- criar politica de designacao por agente
- limitar paralelismo quando houver dependencia narrativa

### Fase 3: Memoria compartilhada mais forte

Objetivo:

- manter unidade argumentativa ao longo do documento

Acoes:

- expandir blackboard para memoria narrativa
- persistir tese, subteses, decisoes e lacunas
- permitir reuso entre execucoes
- registrar "estado do capitulo" antes de escrever cada secao

## 12. Conclusao

O estudo geral aponta que o principal desafio nao e gerar texto tecnico. O principal desafio e coordenar a geracao de varias secoes como partes de um mesmo argumento.

`RevisaoSistematica` ja esta mais proximo da solucao porque possui:

- agentes
- blackboard
- revisao iterativa
- mecanismos de coerencia

Mas ainda precisa de um salto de orquestracao:

- router avancado
- designacao explicita
- memoria narrativa compartilhada
- harmonizacao por capitulo mais forte

`RevisaoTecnica`, por sua vez, e util como base de coleta e avaliacao, mas nao como arquitetura principal para escrita longa e coesa.

Portanto, a linha de evolucao mais promissora para esta tarefa e:

1. manter `RevisaoSistematica` como base principal
2. estudar e formalizar um router avancado
3. fortalecer a memoria compartilhada
4. separar melhor os papeis dos agentes
5. tratar coesao como requisito arquitetural, nao apenas editorial

## 13. Fontes externas consultadas

Ollama:

- API introduction: https://docs.ollama.com/api/introduction
- Generate API: https://docs.ollama.com/api/generate
- Chat API: https://docs.ollama.com/api/chat
- List models: https://docs.ollama.com/api/tags
- List running models: https://docs.ollama.com/api/ps
- Structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- Embeddings API: https://docs.ollama.com/api/embed

Padroes de multiagentes e memoria:

- LangChain multi-agent overview: https://docs.langchain.com/oss/python/langchain/multi-agent
- Supervisor/subagents pattern: https://docs.langchain.com/oss/python/langchain/supervisor
- Handoffs pattern: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence

## 14. Forma curta para apresentar isso na task

Se voce precisar falar de forma simples para professor, lider ou avaliador, pode usar algo assim:

"Os dois repositorios foram revisados. O `RevisaoSistematica` ja possui uma arquitetura mais avancada com multiagentes, blackboard e revisao iterativa, mas ainda apresenta limitacoes de coesao porque a escrita continua muito segmentada por secao e a harmonizacao entra tarde no fluxo. O `RevisaoTecnica` e mais linear e util para coleta e avaliacao, mas nao resolve bem redacao longa e integrada. O estudo indica que a melhor evolucao para o servidor atual e adotar um supervisor com router avancado, designacao explicita de tarefas e memoria compartilhada orientada a narrativa, aproveitando o Ollama local e a GPU disponivel para distribuir papeis entre modelos e agentes sem perder qualidade tecnica." 
