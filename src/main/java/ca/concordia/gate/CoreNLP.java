package ca.concordia.gate;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import gate.*;
import gate.creole.AbstractLanguageAnalyser;
import gate.creole.ExecutionException;
import gate.creole.ResourceInstantiationException;
import gate.creole.metadata.CreoleParameter;
import gate.creole.metadata.CreoleResource;
import gate.creole.metadata.Optional;
import gate.creole.metadata.RunTime;
import gate.util.GateRuntimeException;
import gate.util.InvalidOffsetException;

import java.util.*;
import java.util.stream.Collectors;

/**
 * This plugin runs the CoreNLP pipeline on the document, generating "Token" and "Sentence" annotations.
 */
@CreoleResource(name = "CoreNlpPipeline", comment = "Run CoreNLP pipeline on document")
public class CoreNLP extends AbstractLanguageAnalyser implements ProcessingResource {
    private Boolean useEnhanced;
    private Boolean srParse;
    private Boolean includeTokenizer;
    private Boolean includeSentenceSplitter;
    private Boolean includeParse;

    private String language;

    @Override
    public void reInit() throws ResourceInstantiationException {
        init();
    }

    /**
     * Initialise this resource, and return it.
     */
    @Override
    public Resource init() throws ResourceInstantiationException {
        return super.init();
    }

    public void execute() throws ExecutionException {
        if (this.document == null) throw new GateRuntimeException("No document to process!");
        System.out.println("document name:" + this.document.getName());
        System.out.println("document size" + this.document.getContent().size());
        System.out.println("Language:" + this.language);
        Properties coreNlpProps = new Properties();
        if (language.equals("spanish")) {
            coreNlpProps.setProperty("tokenize.language","es");
            coreNlpProps.setProperty("pos.model","edu/stanford/nlp/models/pos-tagger/spanish-ud.tagger");
            coreNlpProps.setProperty("ner.model","edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz");
            coreNlpProps.setProperty("ner.applyNumericClassifiers","true");
            coreNlpProps.setProperty("ner.useSUTime","true");
            coreNlpProps.setProperty("ner.language","es");
            coreNlpProps.setProperty("sutime.language","spanish");
            coreNlpProps.setProperty("parse.model","edu/stanford/nlp/models/srparser/spanishSR.beam.ser.gz");
            coreNlpProps.setProperty("depparse.model","edu/stanford/nlp/models/parser/nndep/UD_Spanish.gz");
            coreNlpProps.setProperty("depparse.language", "spanish");
        }
        if (srParse) {
            coreNlpProps.setProperty("parse.model", "edu/stanford/nlp/models/srparser/englishSR.beam.ser.gz");
        }
        if (this.includeTokenizer) {
            if (!includeParse) {
                coreNlpProps.setProperty("annotators", "tokenize,ssplit,pos");
                executeWithTokenizerNoParse(coreNlpProps);
            } else {
                coreNlpProps.setProperty("annotators", "tokenize,ssplit,pos,parse,depparse");
                executeWithTokenizer(coreNlpProps);
            }
        } else if (this.includeSentenceSplitter) {
            if (!includeParse) {
                coreNlpProps.setProperty("annotators", "ssplit,pos");
                executeWithoutTokenizerNoParse(coreNlpProps);
            } else {
                coreNlpProps.setProperty("annotators", "ssplit,pos,parse,depparse");
                executeWithoutTokenizer(coreNlpProps);
            }

        } else {
            coreNlpProps.setProperty("annotators", "pos,parse,depparse");
            throw new RuntimeException("Either tokenizer or sentence splitter need to be included");
            //executeWithoutSentenceSplitter(coreNlpProps);
        }
    }

    public void executeWithTokenizer(Properties props) throws ExecutionException {
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        CoreDocument document = new CoreDocument(this.document.getContent().toString());
        pipeline.annotate(document);
        AnnotationSet outputAS = this.document.getAnnotations();
        List<DependencyAnn> dependencyAnnList = new ArrayList<>();
        for (CoreSentence sentence : document.sentences()) {
            // First, add the sentence
            int sentenceStartIndex = sentence.tokens().get(0).beginPosition();
            int sentenceEndIndex = sentence.tokens().get(sentence.tokens().size() - 1).endPosition();
            try {
                outputAS.add((long) sentenceStartIndex, (long) sentenceEndIndex,
                        "Sentence", Factory.newFeatureMap());
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
        }
        // The below offset helps make sure the IDs of our new Token Annotations
        // are unique.
        int sentenceOffset = 0;
        for (Annotation ano : outputAS) {
            if (sentenceOffset < ano.getId()) {
                sentenceOffset = ano.getId();
            }
        }
        sentenceOffset++;
        for (CoreSentence sentence : document.sentences()) {
            SemanticGraph depGraph;
            if (this.useEnhanced) {
                depGraph = sentence.dependencyParse();
            } else {
                depGraph = sentence.coreMap().get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
            }
            for (IndexedWord currWord : depGraph.vertexListSorted()) {
                List<DependencyRelation> dependencies = new ArrayList<>();
                List<IndexedWord> dependants = new ArrayList<>();
                FeatureMap currWordFeatures = Factory.newFeatureMap();
                for (SemanticGraphEdge outEdge : depGraph.outgoingEdgeList(currWord)) {
                    dependencies.add(new DependencyRelation(outEdge.getRelation().toString(),
                            outEdge.getDependent().index() + sentenceOffset));
                    dependants.add(outEdge.getDependent());
                }
                currWordFeatures.put("dependencies", dependencies);
                currWordFeatures.put("length", currWord.originalText().length());
                currWordFeatures.put("string", currWord.originalText());
                currWordFeatures.put("category", sentence.posTags().get(currWord.index() - 1));
                try {
                    outputAS.add(currWord.index() + sentenceOffset, (long) currWord.beginPosition(),
                            (long) currWord.endPosition(), "Token", currWordFeatures);
                } catch (InvalidOffsetException e) {
                    e.printStackTrace();
                }
                // Add dependencies corresponding to the current word
                for (int i = 0; i < dependants.size(); i++) {
                    FeatureMap depFeatures = Factory.newFeatureMap();
                    DependencyAnn depAnn = new DependencyAnn();
                    depAnn.startOffset = currWord.beginPosition();
                    depAnn.endOffset = currWord.endPosition();
                    List<Integer> depArgs = new ArrayList<>();
                    depArgs.add(currWord.index() + sentenceOffset);
                    depArgs.add(dependants.get(i).index() + sentenceOffset);
                    depFeatures.put("args", depArgs);
                    depFeatures.put("kind",dependencies.get(i).getType());
                    depAnn.featureMap = depFeatures;
                    dependencyAnnList.add(depAnn);
                }
            }
            sentenceOffset += depGraph.vertexListSorted().size() + 1;
            Tree constituencyTree = sentence.constituencyParse();
            createToken(constituencyTree, sentence.tokens(), constituencyTree.getLeaves(),
                    constituencyTree.preOrderNodeList(), sentenceOffset, outputAS);
            sentenceOffset += constituencyTree.preOrderNodeList().size() + 1;
        }
        for (DependencyAnn depAnn: dependencyAnnList) {
            try {
                outputAS.add(depAnn.startOffset, depAnn.endOffset, "Dependency", depAnn.featureMap);
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
        }
    }

    public void executeWithTokenizerNoParse(Properties props) throws ExecutionException {
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        CoreDocument document = new CoreDocument(this.document.getContent().toString());
        pipeline.annotate(document);
        AnnotationSet outputAS = this.document.getAnnotations();
        List<DependencyAnn> dependencyAnnList = new ArrayList<>();
        for (CoreSentence sentence : document.sentences()) {
            // First, add the sentence
            int sentenceStartIndex = sentence.tokens().get(0).beginPosition();
            int sentenceEndIndex = sentence.tokens().get(sentence.tokens().size() - 1).endPosition();
            try {
                outputAS.add((long) sentenceStartIndex, (long) sentenceEndIndex,
                        "Sentence", Factory.newFeatureMap());
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
        }
        // The below offset helps make sure the IDs of our new Token Annotations
        // are unique.
        int sentenceOffset = 0;
        for (Annotation ano : outputAS) {
            if (sentenceOffset < ano.getId()) {
                sentenceOffset = ano.getId();
            }
        }
        sentenceOffset++;
        for (CoreSentence sentence : document.sentences()) {
            List<String> posTags = sentence.posTags();
            List<CoreLabel> tokens = sentence.tokens();
            for (int i = 0; i < tokens.size(); i++) {
               CoreLabel token = tokens.get(i);
               String posTag = posTags.get(i);
               FeatureMap tokenFeatures = Factory.newFeatureMap();
               tokenFeatures.put("length",token.size());
               tokenFeatures.put("category",posTag);
               tokenFeatures.put("string", token.word());
               try {
                   outputAS.add((long)token.beginPosition(),
                           (long)token.endPosition(), "Token", tokenFeatures);
               } catch (InvalidOffsetException e) {
                   e.printStackTrace();
               }
            }
        }
    }

    /**
     * A recursive method that walks down the given constituency parse tree of a sentence
     * while generating SyntaxTreeNode annotations.
     *
     * @param node           the root of the current subtree
     * @param tokens         the tokens of the sentence
     * @param leaves         the leaf nodes that correspond to tokens
     * @param nodeOrdering   All the nodes in the tree as a list
     * @param sentenceOffset offset used to give SyntaxTreeNode annotations unique IDs
     * @param outputAS       the output annotation set
     * @return The span of the tree node in the corresponding text it represents.
     */
    public Integer[] createToken(Tree node, List<CoreLabel> tokens, List<Tree> leaves,
                                 List<Tree> nodeOrdering, int sentenceOffset, AnnotationSet outputAS) {
        int nodeID = nodeOrdering.indexOf(node) + sentenceOffset;
        FeatureMap nodeFeatures = Factory.newFeatureMap();
        nodeFeatures.put("cat", node.label().value());
        nodeFeatures.put("ID", nodeID);
        if (node.isLeaf()) {
            int index = leaves.indexOf(node);
            assert index != -1;
            CoreLabel token = tokens.get(index);
//            try {
//                outputAS.add(nodeID, (long) token.beginPosition(),
//                        (long) token.endPosition(), "SyntaxTreeNode", nodeFeatures);
//            } catch (InvalidOffsetException e) {
//                e.printStackTrace();
//            }
            return new Integer[]{token.beginPosition(), token.endPosition()};
        } else {
            Integer minOffset = null;
            Integer maxOffset = null;
            List<Integer> childrenList = new ArrayList<>();
            for (Tree child : node.getChildrenAsList()) {
                int childID = nodeOrdering.indexOf(child) + sentenceOffset;
                if (!child.isLeaf()) { // don't include text nodes as part of SyntaxTreeNode tree
                    childrenList.add(childID);
                }
                Integer[] span = createToken(child, tokens, leaves, nodeOrdering, sentenceOffset, outputAS);
                if ((minOffset == null) || (minOffset > span[0])) {
                    minOffset = span[0];
                }
                if ((maxOffset == null) || (maxOffset < span[1])) {
                    maxOffset = span[1];
                }
            }
            nodeFeatures.put("consists", childrenList);
            try {
                outputAS.add(nodeID, (long) minOffset,
                        (long) maxOffset, "SyntaxTreeNode", nodeFeatures);
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
            return new Integer[]{minOffset, maxOffset};
        }
    }

    private Map<Long,Annotation> posToToken = new HashMap<>();

    private void addTokens(edu.stanford.nlp.pipeline.Annotation document) {
        List<CoreLabel> tokenLabelList = new ArrayList<>();
        for (Annotation ano : this.document.getAnnotations()) {
            // filter for tokens
            if (ano.getType().equals("Token")) {
                String tokenString = (String)ano.getFeatures().get("string");
                CoreLabel tokenLabel = CoreLabel.wordFromString(tokenString);
                tokenLabel.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class,
                        Math.toIntExact(ano.getStartNode().getOffset()));
                tokenLabel.set(CoreAnnotations.CharacterOffsetEndAnnotation.class,
                        Math.toIntExact(ano.getEndNode().getOffset()));
                tokenLabel.set(CoreAnnotations.TextAnnotation.class, tokenString);
                tokenLabel.set(CoreAnnotations.IsNewlineAnnotation.class, false);
                tokenLabel.set(CoreAnnotations.ValueAnnotation.class, tokenString);
                tokenLabelList.add(tokenLabel);
//                System.out.println("Generated Token: " + tokenString + " (" +
//                        ano.getStartNode().getOffset()+ "," + ano.getEndNode().getOffset() + ")");
                assert !this.posToToken.containsKey(ano.getStartNode().getOffset());
                this.posToToken.put(ano.getStartNode().getOffset(), ano);
            }
        }
        tokenLabelList = tokenLabelList.stream().sorted(Comparator
                        .comparingInt(o -> o.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class)))
                .collect(Collectors.toList());
        document.set(CoreAnnotations.TokensAnnotation.class, tokenLabelList);
    }

    public void executeWithoutTokenizer(Properties props) throws ExecutionException {
        edu.stanford.nlp.pipeline.Annotation document = new edu.stanford.nlp.pipeline.Annotation(this.document.getContent().toString());
        addTokens(document);
//        for (CoreLabel label : document.get(CoreAnnotations.TokensAnnotation.class)) {
//            System.out.println(label.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class) + ","
//                    + label.get(CoreAnnotations.CharacterOffsetEndAnnotation.class) + " " + label.word());
//        }
        // we don't tokenize
        props.setProperty("annotators", "ssplit,pos,parse,depparse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props, false);
        pipeline.annotate(document);
        CoreDocument coreDocument = new CoreDocument(document);
        AnnotationSet outputAS = this.document.getAnnotations();
        for (CoreSentence sentence : coreDocument.sentences()) {
            // First, add the sentence
            int sentenceStartIndex = sentence.tokens().get(0).beginPosition();
            int sentenceEndIndex = sentence.tokens().get(sentence.tokens().size() - 1).endPosition();
            try {
                outputAS.add((long) sentenceStartIndex, (long) sentenceEndIndex,
                        "Sentence", Factory.newFeatureMap());
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
        }

        // The below offset helps make sure the IDs of our new Token Annotations
        // are unique.
        int sentenceOffset = 0;
        for (Annotation ano : outputAS) {
            if (sentenceOffset < ano.getId()) {
                sentenceOffset = ano.getId();
            }
        }
        List<DependencyAnn> dependencyAnnList = new ArrayList<>();
        sentenceOffset++;
        for (CoreSentence sentence : coreDocument.sentences()) {
            SemanticGraph depGraph;
            if (this.useEnhanced) {
                depGraph = sentence.dependencyParse();
            } else {
                depGraph = sentence.coreMap().get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
            }
            for (IndexedWord currWord : depGraph.vertexListSorted()) {
                List<DependencyRelation> dependencies = new ArrayList<>();
                FeatureMap currWordFeatures = this.posToToken.get((long)currWord.beginPosition()).getFeatures();
                for (SemanticGraphEdge outEdge : depGraph.outgoingEdgeList(currWord)) {
                    FeatureMap depFeatures = Factory.newFeatureMap();
                    DependencyAnn depAnn = new DependencyAnn();
                    depAnn.startOffset = currWord.beginPosition();
                    depAnn.endOffset = currWord.endPosition();
                    List<Integer> depArgs = new ArrayList<>();
                    depArgs.add(this.posToToken.get((long)currWord.beginPosition()).getId());
                    depArgs.add(this.posToToken.get((long)outEdge.getDependent().beginPosition()).getId());
                    depFeatures.put("args", depArgs);
                    depFeatures.put("kind",outEdge.getRelation().toString());
                    depAnn.featureMap = depFeatures;
                    dependencyAnnList.add(depAnn);
                    dependencies.add(new DependencyRelation(outEdge.getRelation().toString(),
                            this.posToToken.get((long)outEdge.getDependent().beginPosition()).getId()));
                }
                currWordFeatures.put("dependencies", dependencies);
                currWordFeatures.put("length", currWord.originalText().length());
                currWordFeatures.put("string", currWord.originalText());
                currWordFeatures.put("category", sentence.posTags().get(currWord.index() - 1));

            }
            sentenceOffset += depGraph.vertexListSorted().size() + 1;
            Tree constituencyTree = sentence.constituencyParse();
            createSyntaxTreeNode(constituencyTree, sentence.tokens(), constituencyTree.getLeaves(),
                    constituencyTree.preOrderNodeList(), sentenceOffset, outputAS);
            sentenceOffset += constituencyTree.preOrderNodeList().size() + 1;
        }
        for (DependencyAnn depAnn: dependencyAnnList) {
            try {
                outputAS.add(depAnn.startOffset, depAnn.endOffset, "Dependency", depAnn.featureMap);
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
        }
    }


    public void executeWithoutTokenizerNoParse(Properties props) throws ExecutionException {
        edu.stanford.nlp.pipeline.Annotation document = new edu.stanford.nlp.pipeline.Annotation(this.document.getContent().toString());
        addTokens(document);
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props, false);
        pipeline.annotate(document);
        CoreDocument coreDocument = new CoreDocument(document);
        AnnotationSet outputAS = this.document.getAnnotations();
        for (CoreSentence sentence : coreDocument.sentences()) {
            // First, add the sentence
            int sentenceStartIndex = sentence.tokens().get(0).beginPosition();
            int sentenceEndIndex = sentence.tokens().get(sentence.tokens().size() - 1).endPosition();
            try {
                outputAS.add((long) sentenceStartIndex, (long) sentenceEndIndex,
                        "Sentence", Factory.newFeatureMap());
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
            List<CoreLabel> sentenceTokens = sentence.tokens();
            List<String> sentencePosList = sentence.posTags();
            // Update tokens with pos tag
            for (int i = 0; i < sentence.tokens().size(); i++) {
                CoreLabel currToken = sentenceTokens.get(i);
                FeatureMap currTokenFeatures = this.posToToken.get((long)currToken.beginPosition()).getFeatures();
                currTokenFeatures.put("length",currToken.size());
                currTokenFeatures.put("category",sentencePosList.get(i));
                currTokenFeatures.put("string", currToken.word());
            }
        }
    }
//
//    private void addSentencesAndTokens(edu.stanford.nlp.pipeline.Annotation document) {
//        List<CoreLabel> allTokens = new ArrayList<>();
//        List<CoreMap> allSentences = new ArrayList<>();
//        System.out.println("adding sentences");
//        int sentenceIndex = 0;
//        for (Annotation ano : this.document.getAnnotations()) {
//            // filter for tokens
//            if (ano.getType().equals("Sentence")) {
//                ArrayCoreMap sentence = new ArrayCoreMap();
//                Long sentenceStartOffset = ano.getStartNode().getOffset();
//                Long sentenceEndOffset = ano.getEndNode().getOffset();
//                sentence.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class,
//                        Math.toIntExact(sentenceStartOffset));
//                sentence.set(CoreAnnotations.CharacterOffsetEndAnnotation.class,
//                        Math.toIntExact(sentenceEndOffset));
//                sentence.set(CoreAnnotations.SentenceIndexAnnotation.class, sentenceIndex);
//                List<CoreLabel> tokensInSentence = new ArrayList<>();
//                for (Annotation anoInSentence : this.document.getAnnotations().getContained(sentenceStartOffset,sentenceEndOffset)) {
//                    if (anoInSentence.getType().equals("Token")) {
//                        Annotation tokenAno = anoInSentence;
//                        String tokenString = (String)tokenAno.getFeatures().get("string");
//                        CoreLabel tokenLabel = CoreLabel.wordFromString(tokenString);
//                        tokenLabel.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class,
//                                Math.toIntExact(tokenAno.getStartNode().getOffset()));
//                        tokenLabel.set(CoreAnnotations.CharacterOffsetEndAnnotation.class,
//                                Math.toIntExact(tokenAno.getEndNode().getOffset()));
//                        tokenLabel.set(CoreAnnotations.TextAnnotation.class, tokenString);
//                        tokenLabel.set(CoreAnnotations.IsNewlineAnnotation.class, false);
//                        tokenLabel.set(CoreAnnotations.ValueAnnotation.class, tokenString);
//                        tokensInSentence.add(tokenLabel);
//                        assert !this.posToToken.containsKey(tokenAno.getStartNode().getOffset());
//                        this.posToToken.put(tokenAno.getStartNode().getOffset(), ano);
//                    }
//                }
//                tokensInSentence = tokensInSentence.stream().sorted(Comparator
//                                .comparingInt(o -> o.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class)))
//                        .collect(Collectors.toList());
//                sentence.set(CoreAnnotations.TokensAnnotation.class, tokensInSentence);
//                allTokens.addAll(tokensInSentence);
//                sentenceIndex++;
//                allSentences.add(sentence);
//            }
//        }
//        allTokens = allTokens.stream().sorted(Comparator
//                        .comparingInt(o -> o.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class)))
//                .collect(Collectors.toList());
//        document.set(CoreAnnotations.TokensAnnotation.class, allTokens);
//        document.set(CoreAnnotations.SentencesAnnotation.class, allSentences);
//    }
//
//    public void executeWithoutSentenceSplitter(Properties props) throws ExecutionException {
//        edu.stanford.nlp.pipeline.Annotation document = new edu.stanford.nlp.pipeline.Annotation(this.document.getContent().toString());
//        addSentencesAndTokens(document);
//        StanfordCoreNLP pipeline = new StanfordCoreNLP(props, false);
//        pipeline.annotate(document);
//        CoreDocument coreDocument = new CoreDocument(document);
//        AnnotationSet outputAS = this.document.getAnnotations();
//        // The below offset helps make sure the IDs of our new Token Annotations
//        // are unique.
//        int sentenceOffset = 0;
//        for (Annotation ano : outputAS) {
//            if (sentenceOffset < ano.getId()) {
//                sentenceOffset = ano.getId();
//            }
//        }
//        List<DependencyAnn> dependencyAnnList = new ArrayList<>();
//        sentenceOffset++;
//        for (CoreSentence sentence : coreDocument.sentences()) {
//            SemanticGraph depGraph;
//            if (this.useEnhanced) {
//                depGraph = sentence.dependencyParse();
//            } else {
//                depGraph = sentence.coreMap().get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
//            }
//            for (IndexedWord currWord : depGraph.vertexListSorted()) {
//                List<DependencyRelation> dependencies = new ArrayList<>();
//                FeatureMap currWordFeatures = this.posToToken.get((long)currWord.beginPosition()).getFeatures();
//                for (SemanticGraphEdge outEdge : depGraph.outgoingEdgeList(currWord)) {
//                    FeatureMap depFeatures = Factory.newFeatureMap();
//                    DependencyAnn depAnn = new DependencyAnn();
//                    depAnn.startOffset = currWord.beginPosition();
//                    depAnn.endOffset = currWord.endPosition();
//                    List<Integer> depArgs = new ArrayList<>();
//                    depArgs.add(this.posToToken.get((long)currWord.beginPosition()).getId());
//                    depArgs.add(this.posToToken.get((long)outEdge.getDependent().beginPosition()).getId());
//                    depFeatures.put("args", depArgs);
//                    depFeatures.put("kind",outEdge.getRelation().toString());
//                    depAnn.featureMap = depFeatures;
//                    dependencyAnnList.add(depAnn);
//                    dependencies.add(new DependencyRelation(outEdge.getRelation().toString(),
//                            this.posToToken.get((long)outEdge.getDependent().beginPosition()).getId()));
//                }
//                currWordFeatures.put("dependencies", dependencies);
//                currWordFeatures.put("length", currWord.originalText().length());
//                currWordFeatures.put("string", currWord.originalText());
//                currWordFeatures.put("category", sentence.posTags().get(currWord.index() - 1));
//
//            }
//            sentenceOffset += depGraph.vertexListSorted().size() + 1;
//            Tree constituencyTree = sentence.constituencyParse();
//            createSyntaxTreeNode(constituencyTree, sentence.tokens(), constituencyTree.getLeaves(),
//                    constituencyTree.preOrderNodeList(), sentenceOffset, outputAS);
//            sentenceOffset += constituencyTree.preOrderNodeList().size() + 1;
//        }
//        for (DependencyAnn depAnn: dependencyAnnList) {
//            try {
//                outputAS.add(depAnn.startOffset, depAnn.endOffset, "Dependency", depAnn.featureMap);
//            } catch (InvalidOffsetException e) {
//                e.printStackTrace();
//            }
//        }
//    }

    /**
     * A recursive method that walks down the given constituency parse tree of a sentence
     * while generating SyntaxTreeNode annotations.
     * @param node the root of the current subtree
     * @param tokens the tokens of the sentence
     * @param leaves the leaf nodes that correspond to tokens
     * @param nodeOrdering All the nodes in the tree as a list
     * @param sentenceOffset  offset used to give SyntaxTreeNode annotations unique IDs
     * @param outputAS the output annotation set
     * @return The span of the tree node in the corresponding text it represents.
     */
    public Integer[] createSyntaxTreeNode(Tree node, List<CoreLabel> tokens, List<Tree> leaves,
                                          List<Tree> nodeOrdering, int sentenceOffset, AnnotationSet outputAS) {
        int nodeID = nodeOrdering.indexOf(node) + sentenceOffset;
        FeatureMap nodeFeatures = Factory.newFeatureMap();
        nodeFeatures.put("cat", node.label().value());
        nodeFeatures.put("ID", nodeID);
        if (node.isLeaf()) {
            int index = leaves.indexOf(node);
            assert index != -1;
            CoreLabel token = tokens.get(index);
            try {
                outputAS.add(nodeID, (long) token.beginPosition(),
                        (long) token.endPosition(), "SyntaxTreeNode", nodeFeatures);
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
            return new Integer[]{token.beginPosition(), token.endPosition()};
        } else {
            Integer minOffset = null;
            Integer maxOffset = null;
            List<Integer> childrenList = new ArrayList<>();
            for (Tree child : node.getChildrenAsList()) {
                int childID = nodeOrdering.indexOf(child) + sentenceOffset;
                childrenList.add(childID);
                Integer[] span = createSyntaxTreeNode(child, tokens, leaves, nodeOrdering, sentenceOffset, outputAS);
                if ((minOffset == null) || (minOffset > span[0])) {
                    minOffset = span[0];
                }
                if ((maxOffset == null) || (maxOffset < span[1])) {
                    maxOffset = span[1];
                }
            }
            nodeFeatures.put("consists", childrenList);
            try {
                outputAS.add(nodeID, (long) minOffset,
                        (long) maxOffset, "SyntaxTreeNode", nodeFeatures);
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
            return new Integer[]{minOffset, maxOffset};
        }
    }

    @RunTime
    @Optional
    @CreoleParameter(comment = "If true, enhanced dependencies will be generated", defaultValue = "false")
    public void setUseEnhanced(Boolean useEnhanced) {
        this.useEnhanced = useEnhanced;
    }

    @RunTime
    @Optional
    @CreoleParameter(comment = "If true, use the shift-reduce parser for constituency parse", defaultValue = "false")
    public void setSrParse(Boolean srParse) {
        this.srParse = srParse;
    }

    @RunTime
    @Optional
    @CreoleParameter(comment = "If true, tokenize the input. If false, the user should provide tokenized input", defaultValue = "true")
    public void setIncludeTokenizer(Boolean includeTokenizer) {
        this.includeTokenizer = includeTokenizer;
    }

    @RunTime
    @Optional
    @CreoleParameter(comment = "If true, tokenize the input. If false, the user should provide tokenized input", defaultValue = "true")
    public void setIncludeSentenceSplitter(Boolean includeSentenceSplitter) {
        this.includeSentenceSplitter = includeSentenceSplitter;
    }

    @RunTime
    @Optional
    @CreoleParameter(comment = "If true, run dependency and constituency parsers on input.", defaultValue = "true")
    public void setIncludeParse(Boolean includeParse) {
        this.includeParse = includeParse;
    }


    @RunTime
    @Optional
    @CreoleParameter(comment = "The language of the input text", defaultValue = "english")
    public void setLanguage(String language) {
        this.language = language;
    }

    public Boolean getIncludeTokenizer() {return this.includeTokenizer;}

    public Boolean getIncludeSentenceSplitter() {return this.includeSentenceSplitter;}

    public Boolean getUseEnhanced() {
        return this.useEnhanced;
    }

    public Boolean getSrParse() {
        return this.srParse;
    }

    public String getLanguage() {
        return this.language;
    }

    public Boolean getIncludeParse() {
        return this.includeParse;
    }
}