package ca.concordia.gate;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.Tree;
import gate.*;
import gate.creole.AbstractLanguageAnalyser;
import gate.creole.ExecutionException;
import gate.creole.ResourceInstantiationException;
import gate.creole.metadata.CreoleResource;
import gate.util.GateRuntimeException;
import gate.util.InvalidOffsetException;

import java.util.*;
import java.util.stream.Collectors;

@CreoleResource(name = "New Stanford Parser - tokenizer", comment = "Run CoreNLP pipeline without the tokenizer")
public class CoreNlpWithoutTokenizer extends AbstractLanguageAnalyser implements ProcessingResource {
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
                System.out.println("Generated Token: " + tokenString + " (" +
                        ano.getStartNode().getOffset()+ "," + ano.getEndNode().getOffset() + ")");
                assert !this.posToToken.containsKey(ano.getStartNode().getOffset());
                this.posToToken.put(ano.getStartNode().getOffset(), ano);
            }
        }
        tokenLabelList = tokenLabelList.stream().sorted(Comparator
                .comparingInt(o -> o.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class)))
                .collect(Collectors.toList());
        document.set(CoreAnnotations.TokensAnnotation.class, tokenLabelList);
    }

    public void execute() throws ExecutionException {
        if (this.document == null)
            throw new GateRuntimeException("No document to process!");
        System.out.println("document name:" + this.document.getName());
        System.out.println("document size" + this.document.getContent().size());
        edu.stanford.nlp.pipeline.Annotation document = new edu.stanford.nlp.pipeline.Annotation(this.document.getContent().toString());
        addTokens(document);
        for (CoreLabel label : document.get(CoreAnnotations.TokensAnnotation.class)) {
            System.out.println(label.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class) + ","
                    + label.get(CoreAnnotations.CharacterOffsetEndAnnotation.class) + " " + label.word());
        }
        Properties props = new Properties();
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
            SemanticGraph depGraph = sentence.dependencyParse();
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
}
