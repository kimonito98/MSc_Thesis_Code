/*    Copyright (c) 2010-2012 Delft University of Technology.
 *
 *    This software is protected by national and international copyright.
 *    Any unauthorized use, reproduction or modification is unlawful and
 *    will be prosecuted. Commercial and non-private application of the
 *    software in any form is strictly prohibited unless otherwise granted
 *    by the authors.
 *
 *    The code is provided without any warranty; without even the implied
 *    warranty of merchantibility or fitness for a particular purpose.
 *
 *    Changelog
 *      YYMMDD    Author            Comment
 *      102511    D. Dirkx          First version of file.
 *      110120    D. Dirkx          Finalized for code check.
 *      110208    K. Kumar          Updated file header; correct Doxygen comments; minor changes
 *                                  to functions.
 *      110209    D. Dirkx          Minor changes.
 *      110905    S. Billemont      Reorganized includes.
 *                                  Moved (con/de)structors and getter/setters to header.
 *
 *    References
 *
 */

#ifndef TUDAT_TORUS_H
#define TUDAT_TORUS_H

#include <Eigen/Core>
#include "Tudat/Mathematics/GeometricShapes/singleSurfaceGeometry.h"

namespace tudat
{

//! Torus class.
/*!
 * Class that defines the torus shape. The parameters are the majorRadius_,
 * denoting the distance from the center of the torus to the center of the tube
 * and the tubeRadius_, denoting the radius of the tube. Independent variables
 * are the major and minor circumferential angles, with the former denoting
 * the angle by which the tube is 'revolved' and the latter denoting the point
 * on the circular cross-section of the tube.
 */
class Torus: public SingleSurfaceGeometry
{
public:

    //! Default constructor.
    /*!
     *  Default constructor.
     */
    Torus( ) : majorRadius_( -0.0 ), minorRadius_( -0.0 ) { }

    //! Get surface point on torus.
    /*!
     * Retrieves a surface point in Cartesian coordinates on the torus from
     * values of the two independent variables.
     * Function uses cartesianPositionVector_ member variable. Values of
     * this vector set previous to function call are irrelevant.
     * \param majorCircumferentialAngle Major circumferential angle of the
     *         torus at which to retrieve the surface point.
     * \param minorCircumferentialAngle Minor circumferential angle of the
     *         torus at which to retrieve the surface point.
     * \return Point on torus in Cartesian coordinates.
     */
    Eigen::VectorXd getSurfacePoint( double majorCircumferentialAngle,
                                     double minorCircumferentialAngle );

    //! Get surface derivative on torus.
    /*!
     * Retrieves the derivatives of the surface point with respect to the two
     * independent variables. For powerOfMajorCircumferentialAngle = 1 and
     * powerOfMinorCircumferentialAngle = 2 the function returns:
     * \f[
     *      \frac{ d^{ 3 } ( x, y, z ) } { du * dv^{ 2 } }
     * \f]
     * (with u and v the major and minor angles).
     * \param majorCircumferentialAngle Major circumferential angle.
     * \param minorCircumferentialAngle Minor circumferential angle.
     * \param powerOfMajorCircumferentialAngleDerivative Power of derivative
     *          with respect to major circumferential angle.
     * \param powerOfMinorCircumferentialAngleDerivative Power of the derivative
     *          with respect to the minor circumferential angle.
     * \return Surface derivative on torus.
     */
    Eigen::VectorXd getSurfaceDerivative( double majorCircumferentialAngle,
                                          double minorCircumferentialAngle,
                                          int powerOfMajorCircumferentialAngleDerivative,
                                          int powerOfMinorCircumferentialAngleDerivative );

    //! Get parameter of torus.
    /*!
     * Retrieves a parameter of the torus.
     * Function uses parameter_ member variable to prevent multiple
     * declarations.
     * \param index Index of parameter to return ( index = 0: returns major
     *          radius; index = 1: returns minor radius ).
     * \return Selected parameter.
     */
    double getParameter( int index );

    //! Set parameter of torus.
    /*!
     * Sets a parameter of the torus.
     * \param index Index of parameter to return ( index = 0: returns major
     *          radius; index = 1: returns minor radius ).
     * \param parameter Value of parameter to set.
     */
    void setParameter( int index, double parameter );

    //! Get major radius.
    /*!
     * Returns the major radius.
     * \return Major radius.
     */
    double& getMajorRadius( ) { return majorRadius_; }

    //! Set major radius.
    /*!
     * Sets the major radius.
     * \param majorRadius Major radius.
     */
    void setMajorRadius( double majorRadius ) { majorRadius_ = majorRadius; }

    //! Get minor radius.
    /*!
     * Returns the minor radius.
     * \return Minor radius.
     */
    double& getMinorRadius( ) { return minorRadius_; }

    //! Set minor radius.
    /*!
     * Sets the minor radius.
     * \param minorRadius Minor radius.
     */
    void setMinorRadius( double minorRadius ) { minorRadius_ = minorRadius; }

    //! Get maximum of major circumferential angle.
    /*!
     * Returns the maximum value of the major circumferential angle.
     * \return Maximum value of major circumferential angle.
     */
    double getMaximumMajorCircumferentialAngle( ) { return getMaximumIndependentVariable( 1 ); }

    //! Get maximum of minor circumferential angle.
    /*!
     * Returns the maximum of the minor circumferential angle.
     * \return Maximum minor circumferential angle.
     */
    double getMaximumMinorCircumferentialAngle( ) { return getMaximumIndependentVariable( 2 ); }

    //! Get minimum of major circumferential angle.
    /*!
     * Returns the minimum value of the major circumferential angle.
     * \return Minimum value of major circumferential angle.
     */
    double getMinimumMajorCircumferentialAngle( ) { return getMinimumIndependentVariable( 1 ); }

    //! Get minimum of minor circumferential angle.
    /*!
     * Returns the minimum value of the minor circumferential angle.
     * \return Minimum value of minor circumferential angle.
     */
    double getMinimumMinorCircumferentialAngle( ) { return getMinimumIndependentVariable( 2 ); }

    //! Set maximum of major circumferential angle.
    /*!
     * Sets the maximum value of the major circumferential angle.
     * \param maximumMajorCircumferentialAngle Maximum value of major
     *          circumferential angle.
     */
    void setMaximumMajorCircumferentialAngle( double maximumMajorCircumferentialAngle )
    { setMaximumIndependentVariable( 1, maximumMajorCircumferentialAngle ); }

    //! Set minimum of major circumferential angle.
    /*!
     * Sets the minimum value of the major circumferential angle.
     * \param minimumMajorCircumferentialAngle Minimum value of major
     *          circumferential angle.
     */
    void setMinimumMajorCircumferentialAngle( double minimumMajorCircumferentialAngle )
    { setMinimumIndependentVariable( 1, minimumMajorCircumferentialAngle ); }

    //! Set maximum of minor circumferential angle.
    /*!
     * Sets the maximum value of the minor circumferential angle.
     * \param maximumMinorCircumferentialAngle Maximum value of minor
     *          circumferential angle.
     */
    void setMaximumMinorCircumferentialAngle( double maximumMinorCircumferentialAngle )
    { setMaximumIndependentVariable( 2, maximumMinorCircumferentialAngle ); }

    //! Set minimum of minor circumferential angle.
    /*!
     * Sets the minimum value of the minor circumferential angle.
     * \param minimumMinorCircumferentialAngle Minimum value of minor
     *          circumferential angle.
     */
    void setMinimumMinorCircumferentialAngle( double minimumMinorCircumferentialAngle ) 
    { setMinimumIndependentVariable( 2, minimumMinorCircumferentialAngle ); }

    //! Overload ostream to print class information.
    /*!
     * Overloads ostream to print class information, prints the class type,
     * the ranges for the minor and major circumferential angles, and the
     * major and minor radii.
     */
    friend std::ostream &operator<<( std::ostream &stream, Torus& torus );

private:

    //! Major radius.
    /*!
     * Major radius, i.e. radius of center of torus to center of tube.
     */
    double majorRadius_;

    //! Minor radius.
    /*!
     * Minor radius, i.e. radius of tube.
     */
    double minorRadius_;
};

} // namespace tudat

#endif // TUDAT_TORUS_H