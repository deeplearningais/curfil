#if 0
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#endif
#ifndef CURFIL_EXPORT_HPP
#define CURFIL_EXPORT_HPP

#include <boost/property_tree/ptree.hpp>
#include <string>

#include "random_forest_image.h"

namespace curfil {

/**
 * Helper class to export a random tree or random forest to disk in compressed (gzip) JSON format.
 * @ingroup import_export_trees
 *
 * @see RandomTreeImport
 */
class RandomTreeExport {


public:

	/**
	 * Prepare an export with the given configuration.
	 *
	 * @param configuration the configuration the random tree/forest was trained with
	 * @param outputFolder the folder the JSON file(s) should be written to
	 * @param trainingFolder the folder the training images were loaded from
	 * @param verbose if true, the JSON file contains per-node profiling information. Attention: verbose JSON files are significantly larger even if compressed.
	 */
    RandomTreeExport(const TrainingConfiguration& configuration, const std::string& outputFolder,
            const std::string& trainingFolder, bool verbose);

    /**
     * Export the given random tree to disk as compressed (gzip) JSON file.
     *
     * @param tree the random tree which is usually part of a random forest
     * @param treeNr the number (id) of the tree in the random forest. Use 0 if the tree is not part of a forest.
     */
    void writeJSON(const RandomTreeImage& tree, size_t treeNr) const;

    /**
     * Export the given random forest to disk as compressed (gzip) JSON files.
     * Each tree of the forest is stored in a separate file.
     *
     * @param ensemble the random forest that contains several random trees
     */
    template<class TreeEnsemble>
    void writeJSON(const TreeEnsemble& ensemble) const {

        CURFIL_INFO("writing tree files to " << outputFolder << " (verbose: " << verbose << ")");

        for (size_t treeNr = 0; treeNr < ensemble.getTrees().size(); treeNr++) {
            writeJSON(*(ensemble.getTree(treeNr)), treeNr);
        }

        CURFIL_INFO("wrote JSON files to " << outputFolder);
    }

private:
    static std::string spaces(int level);

    static void writeXY(boost::property_tree::ptree& pt, const std::string& name, const XY& xy);

    static void writeFeatureDetails(boost::property_tree::ptree& pt, const ImageFeatureFunction& feature);

    static std::vector<std::map<std::string, std::string> > parseProcCpuInfo();

    static int getPhysicalCores();

    static int getPhysicalCpus();

    template<class T>
    static boost::property_tree::ptree toPropertyTree(const std::vector<T>& input) {
        boost::property_tree::ptree propertyTree;
        for (const auto& v : input) {
            propertyTree.push_back(std::make_pair("", boost::lexical_cast<std::string>(v)));
        }
        return propertyTree;
    }

    static boost::property_tree::ptree getProcessorModelNames();

    void writeTree(boost::property_tree::ptree& pt, const RandomTree<PixelInstance, ImageFeatureFunction>& tree) const;

    void writeTree(boost::property_tree::ptree& pt, const RandomTreeImage& tree) const;

private:

    const boost::posix_time::ptime date;
    const TrainingConfiguration& configuration;
    const std::string outputFolder;
    const std::string trainingFolder;
    const bool verbose;

};

}

#endif
