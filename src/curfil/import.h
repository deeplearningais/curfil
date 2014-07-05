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
