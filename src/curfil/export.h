#ifndef CURFIL_EXPORT_HPP
#define CURFIL_EXPORT_HPP

#include <boost/property_tree/ptree.hpp>
#include <string>

#include "random_forest_image.h"

namespace curfil {

class RandomTreeExport {

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

public:

    RandomTreeExport(const TrainingConfiguration& configuration, const std::string& outputFolder,
            const std::string& trainingFolder, bool verbose);

    void writeJSON(const RandomTreeImage& tree, size_t treeNr) const;

    template<class TreeEnsemble>
    void writeJSON(const TreeEnsemble& ensemble) const {

        CURFIL_INFO("writing tree files to " << outputFolder << " (verbose: " << verbose << ")");

        for (size_t treeNr = 0; treeNr < ensemble.getTrees().size(); treeNr++) {
            writeJSON(*(ensemble.getTree(treeNr)), treeNr);
        }

        CURFIL_INFO("wrote JSON files to " << outputFolder);
    }

private:

    const boost::posix_time::ptime date;
    const TrainingConfiguration& configuration;
    const std::string outputFolder;
    const std::string trainingFolder;
    const bool verbose;

};

}

#endif
