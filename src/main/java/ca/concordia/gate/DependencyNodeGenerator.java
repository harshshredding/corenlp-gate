package ca.concordia.gate;

import gate.*;
import gate.creole.AbstractLanguageAnalyser;
import gate.creole.ExecutionException;
import gate.creole.ResourceInstantiationException;
import gate.creole.metadata.CreoleResource;
import gate.util.GateRuntimeException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This plugin generates the DependencyTreeNodes required by the DependencyTreeViewer plugin
 */
@CreoleResource(name = "Dependency Node Generator", comment = "generates dependency nodes for dependency viewer")
public class DependencyNodeGenerator extends AbstractLanguageAnalyser {

    @Override
    public void reInit() throws ResourceInstantiationException
    {
        init();
    }

    /** Initialise this resource, and return it. */
    @Override
    public Resource init() throws ResourceInstantiationException
    {
        return super.init();
    }

    public void execute() throws ExecutionException {
        if(this.document == null)
            throw new GateRuntimeException("No document to process!");
        System.out.println("document name :" + this.document.getName());
        System.out.println("no of annotations : " + this.document.getAnnotations().size());
        // Map to prepare dependency node information. We have one dependency node per token.
        Map<Integer, FeatureMap> tokenToDepNode = new HashMap<>();
        // Make sure that each generated DependencyTreeNode will have a unique ID.
        // We do this by labeling our dependency nodes with Ids greater than the current
        // max annotation id.
        int uniqueID = 0;
        for (Annotation ano: this.document.getAnnotations()) {
            if (uniqueID < ano.getId()) {
                uniqueID = ano.getId();
            }
        }
        uniqueID++;
        // populate the dependency information
        for (Annotation ano : this.document.getAnnotations()) {
            // filter for tokens
            if (ano.getType().equals("Token")) {
                Annotation tokenAno = ano;
                // DEBUG : System.out.println("Token found");
                int tokenId = tokenAno.getId();
                FeatureMap currTokenFeats = Factory.newFeatureMap();
                if (!tokenToDepNode.containsKey(tokenId)) {
                    currTokenFeats.put("ID", uniqueID);
                    // Default label for each node is ROOT
                    currTokenFeats.put("cat", "ROOT");
                    uniqueID++;
                } else {
                    currTokenFeats = tokenToDepNode.get(tokenId);
                }
                currTokenFeats.put("startNode", tokenAno.getStartNode());
                currTokenFeats.put("endNode", tokenAno.getEndNode());
                currTokenFeats.put("TokenID", tokenId);
                List<DependencyRelation> dependencies = (List<DependencyRelation>)tokenAno.getFeatures().get("dependencies");
                List<Integer> depIDList = new ArrayList<>();
                if (dependencies != null) {
                    for (int i = 0; i < dependencies.size(); i++) {
                        DependencyRelation dep = dependencies.get(i);
                        Integer depID = dep.getTargetId();
                        String depType = dep.getType();
                        FeatureMap dependentFeats = Factory.newFeatureMap();
                        if (!tokenToDepNode.containsKey(depID)) {
                            dependentFeats.put("ID", uniqueID);
                            uniqueID++;
                        } else {
                            dependentFeats = tokenToDepNode.get(depID);
                        }
                        dependentFeats.put("cat", depType);
                        depIDList.add((Integer)dependentFeats.get("ID"));
                        tokenToDepNode.put(depID, dependentFeats);
                    }
                }
                if (!depIDList.isEmpty()) {
                    currTokenFeats.put("consists", depIDList);
                }
                tokenToDepNode.put(tokenId, currTokenFeats);
            }
        }
        // Ouput the dependencyNodes we have found
        AnnotationSet outputAS = this.document.getAnnotations();
        for (FeatureMap feats : tokenToDepNode.values()) {
            if ((feats.get("startNode") != null) && (feats.get("endNode") != null)) {
                Long startOffset = ((Node)feats.get("startNode")).getOffset();
                Long endOffset = ((Node)feats.get("endNode")).getOffset();
                try {
                    outputAS.add((Integer)feats.get("ID"), startOffset, endOffset, "DependencyTreeNode", feats);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } else {
                System.out.println("we found null");
            }
        }
    }
}
