//
// Created by Michael Plumaris on 21/01/2022.
//

#ifndef TUDATBUNDLE_COMPTONWAVELENGTHACCELERATIONPARTIAL_H
#define TUDATBUNDLE_COMPTONWAVELENGTHACCELERATIONPARTIAL_H


// Michael

#include <memory>

#include "tudat/astro/orbit_determination/acceleration_partials/accelerationPartial.h"
#include "tudat/astro/orbit_determination/estimatable_parameters/comptonWavelength.h"

namespace tudat
{


namespace acceleration_partials
{

Eigen::Vector3d computePartialOfCentralGravityComptonWrtComptonWavelength(
        const Eigen::Vector3d& acceleratedBodyPosition,
        const Eigen::Vector3d& acceleratingBodyPosition,
        const double gravitationalParameter );


Eigen::Matrix3d calculatePartialOfCentralGravityComptonWrtPositionOfAcceleratedBody(
        const Eigen::Vector3d& acceleratedBodyPosition,
        const Eigen::Vector3d& acceleratingBodyPosition,
        const double gravitationalParameter,
        const double ComptonWavelength = std );


class comptonWavelengthAccelerationPartial: public AccelerationPartial
{
public:

    //! Constructor.
    comptonWavelengthAccelerationPartial(
            const std::shared_ptr< gravitation::CentralGravitationalAccelerationModel3d > gravitationalAcceleration,
            const std::string acceleratedBody,
            const std::string acceleratingBody  );

//    //! Destructor.
//    ~comptonWavelengthAccelerationPartial( ){ }

    void wrtPositionOfAcceleratedBody( Eigen::Block< Eigen::MatrixXd > partialMatrix,
                                       const bool addContribution = 1, const int startRow = 0, const int startColumn = 0 )
    {
        if( addContribution )
        {
            partialMatrix.block( startRow, startColumn, 3, 3 ) += currentPartialWrtPosition_;
        }
        else
        {
            partialMatrix.block( startRow, startColumn, 3, 3 ) -= currentPartialWrtPosition_;
        }
    }

    void wrtPositionOfAcceleratingBody( Eigen::Block< Eigen::MatrixXd > partialMatrix,
                                        const bool addContribution = 1, const int startRow = 0, const int startColumn = 0 )
    {
        if( addContribution )
        {
            partialMatrix.block( startRow, startColumn, 3, 3 ) -= currentPartialWrtPosition_;
        }
        else
        {
            partialMatrix.block( startRow, startColumn, 3, 3 ) += currentPartialWrtPosition_;
        }
    }

//    void wrtNonTranslationalStateOfAdditionalBody(
//            Eigen::Block< Eigen::MatrixXd > partialMatrix,
//            const std::pair< std::string, std::string >& stateReferencePoint,
//            const propagators::IntegratedStateType integratedStateType,
//            const bool addContribution = true )
//    {
//        if( stateReferencePoint.first == acceleratedBody_ && integratedStateType == propagators::body_mass_state )
//        {
//            partialMatrix.block( 0, 0, 3, 1 ) +=
//                    ( addContribution ? 1.0 : -1.0 ) * ( radiationPressureFunction_( ) * areaFunction_( ) * radiationPressureCoefficientFunction_( ) *
//                                                         ( sourceBodyState_( ) - acceleratedBodyState_( ) ).normalized( ) /
//                                                         ( acceleratedBodyMassFunction_( ) * acceleratedBodyMassFunction_( ) ) );
//        }
//    }

    bool isStateDerivativeDependentOnIntegratedAdditionalStateTypes(
            const std::pair< std::string, std::string >& stateReferencePoint,
            const propagators::IntegratedStateType integratedStateType )
    {
        if( ( ( stateReferencePoint.first == acceleratingBody_ ||
                ( stateReferencePoint.first == acceleratedBody_  && accelerationUsesMutualAttraction_ ) )
              && integratedStateType == propagators::body_mass_state ) )
        {
            std::cerr<<"Warning, dependency of central gravity on body masses not yet implemented"<<std::endl;
        }
        return 0;
    }


    std::pair< std::function< void( Eigen::MatrixXd& ) >, int >
    getParameterPartialFunction(
            std::shared_ptr< estimatable_parameters::EstimatableParameter< double > > parameter );


    std::pair< std::function< void( Eigen::MatrixXd& ) >, int >
    getParameterPartialFunction(
            std::shared_ptr< estimatable_parameters::EstimatableParameter< Eigen::VectorXd > > parameter );



    //! Function for updating partial w.r.t. the bodies' positions
    /*!
     *  Function for updating common blocks of partial to current state. For the radiation pressure acceleration,
     *  position partial is computed and set.
     *  \param currentTime Time at which partials are to be calculated
     */
    void update( const double currentTime = TUDAT_NAN )
    {
        accelerationUpdateFunction_( currentTime );

        if( !( currentTime_ == currentTime ) )
        {
            acceleratedBodyState_( currentAcceleratedBodyState_ );
            centralBodyState_( currentCentralBodyState_ );
            currentGravitationalParameter_ = gravitationalParameterFunction_( );
            currentComptonWavelength_ = comptonWavelengthFunction_( );

            currentPartialWrtPosition_ = calculatePartialOfCentralGravityComptonWrtPositionOfAcceleratedBody(
                    currentAcceleratedBodyState_,
                    currentCentralBodyState_,
                    currentGravitationalParameter_,
                    currentComptonWavelength_);

            currentTime_ = currentTime;
        }
    }


protected:


    std::pair< std::function< void( Eigen::MatrixXd& ) >, int > getComptonWavelengthPartialFunction(
            const estimatable_parameters::EstimatebleParameterIdentifier& parameterId );

    //! Function to calculate central gravity partial w.r.t. central body gravitational parameter.
    void wrtComptonWavelengthOfCentralBody( Eigen::MatrixXd& comptonWavelengthPartial );

    //! Function to retrieve current gravitational parameter of central body.
    std::function< double( ) > gravitationalParameterFunction_;


    std::function< double( ) > comptonWavelengthFunction_;

    //! Function to retrieve current state of body exerting acceleration.
    std::function< void( Eigen::Vector3d& ) > centralBodyState_;

    //! Function to retrieve current state of body undergoing acceleration.
    std::function< void( Eigen::Vector3d& ) > acceleratedBodyState_;

    //! Boolean denoting whether the gravitational attraction of the central body on the accelerated body is included.
    bool accelerationUsesMutualAttraction_;

    //! Current state of the body undergoing the acceleration (as set by update function).
    Eigen::Vector3d currentAcceleratedBodyState_;

    //! Current state of the body exerting the acceleration (as set by update function).
    Eigen::Vector3d currentCentralBodyState_;

    double currentGravitationalParameter_;

    double currentComptonWavelength_;

    //! Current partial of central gravity acceleration w.r.t. position of body undergoing acceleration
    /*!
     *  Current partial of central gravity acceleration w.r.t. position of body undergoing acceleration
     * ( = -partial of central gravity acceleration w.r.t. position of body exerting acceleration),
     *  calculated and set by update( ) function.
     */
    Eigen::Matrix3d currentPartialWrtPosition_;

    //! Function to update the gravitational acceleration model.
    std::function< void( const double ) > accelerationUpdateFunction_;
};

}






}


#endif //TUDATBUNDLE_COMPTONWAVELENGTHACCELERATIONPARTIAL_H
