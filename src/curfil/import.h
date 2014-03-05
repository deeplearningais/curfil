#ifndef CURFIL_IMPORT_HPP
#define CURFIL_IMPORT_HPP

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <string>

#include "random_forest_image.h"

namespace curfil {

/**
 * Helper class to import a random tree or random forest from disk in compressed (gzip) JSON format.
 * @ingroup import_export_trees
 *
 * @see RandomTreeExport
 */
class RandomTreeImport {


public:

	/**
	 * load (deserialize) a random tree from disk, stored in compressed JSON format
	 */
    static TrainingConfiguration readJSON(const std::string& filename, boost::shared_ptr<RandomTreeImage>& tree,
            std::string& hostname,
            boost::filesystem::path& folderTraining,
            boost::posix_time::ptime& date);

private:

    static XY readXY(const boost::property_tree::ptree& pt);

    static SplitFunction<PixelInstance, ImageFeatureFunction> parseSplit(const boost::property_tree::ptree& pt);

    static boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > readTree(
            const boost::property_tree::ptree& pt,
            const boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >& parent = boost::shared_ptr<
                    RandomTree<PixelInstance, ImageFeatureFunction> >());

    template<class T>
    static std::vector<T> fromPropertyTree(const boost::optional<boost::property_tree::ptree&>& propertyTree,
            const std::vector<T> defaultValue = std::vector<T>()) {

        std::vector<T> values = defaultValue;

        if (propertyTree) {
            boost::property_tree::ptree::const_iterator it;
            for (it = propertyTree.get().begin(); it != propertyTree.get().end(); it++) {
                values.push_back(it->second.get_value<T>());
            }
        }

        return values;
    }

    static cuv::ndarray<WeightType, cuv::host_memory_space> readClassLabelPriorDistribution(
            const boost::property_tree::ptree& p);

};

}

#endif
